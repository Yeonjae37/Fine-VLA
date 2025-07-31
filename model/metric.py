"""Module for computing performance metrics

검색 평가
t2v_result = t2v_metrics(sims, mask)
v2t_result = v2t_metrics(sims, mask)

분류 평가
accuracy = acc(predictions, targets)
top3_acc = my_metric2(predictions, targets, k=3)
mAP = mean_average_precision(predictions, targets)

정렬 평가
precision = video_precision(output, target)
adj_precision = video_precision_adj(output, target)
"""
from pathlib import Path

import numpy as np
import scipy.stats
import torch


def t2v_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing queries from the dataset (two videos
             in MSRVTT only have 19, rather than 20 captions)

    Returns:
        (dict[str:float]): retrieval metrics
    """
    assert sims.ndim == 2, "expected a matrix" #2차원 행렬인지 확인
    num_queries, num_vids = sims.shape
    dists = -sims # 유사도를 거리로 변환 (높은 유사도 = 낮은 거리)
    sorted_dists = np.sort(dists, axis=1) #각 행에 대해 거리를 오름차순으로 정렬

    # The indices are computed such that they slice out the ground truth distances
    # from the psuedo-rectangular dist matrix
    queries_per_video = num_queries // num_vids # 비디오당 텍스트 개수
    gt_idx = [[np.ravel_multi_index([ii, jj], (num_queries, num_vids))
               for ii in range(jj * queries_per_video, (jj + 1) * queries_per_video)]
              for jj in range(num_vids)]
    gt_idx = np.array(gt_idx)
    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)] #정답쌍 위치의 거리값들만 추출
    gt_dists = gt_dists[:, np.newaxis] # 2차원 행렬로 변환
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # --------------------------------
    # NOTE: Breaking ties
    # --------------------------------
    # We sometimes need to break ties (in general, these should occur extremely rarely,
    # but there are pathological cases when they can distort the scores, such as when
    # the similarity matrix is all zeros). Previous implementations (e.g. the t2i
    # evaluation function used
    # here: https://github.com/niluthpol/multimodal_vtt/blob/master/evaluation.py and
    # here: https://github.com/linxd5/VSE_Pytorch/blob/master/evaluation.py#L87) generally
    # break ties "optimistically".  However, if the similarity matrix is constant this
    # can evaluate to a perfect ranking. A principled option is to average over all
    # possible partial orderings implied by the ties. See # this paper for a discussion:
    #    McSherry, Frank, and Marc Najork,
    #    "Computing information retrieval performance measures efficiently in the presence
    #    of tied scores." European conference on information retrieval. Springer, Berlin,
    #    Heidelberg, 2008.
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.8892&rep=rep1&type=pdf

    break_ties = "optimistically"
    #break_ties = "averaging"

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            _, idx = np.unique(rows, return_index=True)
            cols = cols[idx]
        elif break_ties == "averaging":
            # fast implementation, based on this code:
            # https://stackoverflow.com/a/49239335
            locs = np.argwhere((sorted_dists - gt_dists) == 0)

            # Find the split indices
            steps = np.diff(locs[:, 0])
            splits = np.nonzero(steps)[0] + 1
            splits = np.insert(splits, 0, 0)

            # Compute the result columns
            summed_cols = np.add.reduceat(locs[:, 1], splits)
            counts = np.diff(np.append(splits, locs.shape[0]))
            avg_cols = summed_cols / counts
            if False:
                print("Running slower code to verify rank averaging across ties")
                # slow, but more interpretable version, used for testing
                avg_cols_slow = [np.mean(cols[rows == idx]) for idx in range(num_queries)]
                assert np.array_equal(avg_cols, avg_cols_slow), "slow vs fast difference"
                print("passed num check")
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    if cols.size != num_queries:
        import ipdb;
        ipdb.set_trace()
    assert cols.size == num_queries, msg

    if False:
        # overload mask to check that we can recover the scores for single-query
        # retrieval
        print("DEBUGGING MODE")
        query_masks = np.zeros_like(query_masks)
        query_masks[:, 0] = 1  # recover single query score

    if query_masks is not None: # 일부 비디오가 예상보다 적은 캡션을 가진 경우에 대한 처리
        # remove invalid queries
        assert query_masks.size == num_queries, "invalid query mask shape"
        cols = cols[query_masks.reshape(-1).astype(np.bool)]
        assert cols.size == query_masks.sum(), "masking was not applied correctly"
        # update number of queries to account for those that were missing
        num_queries = query_masks.sum()

    if False:
        # sanity check against old logic for square matrices
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        _, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        assert np.array_equal(cols_old, cols), "new metric doesn't match"

    return cols2metrics(cols, num_queries)


def v2t_metrics(sims, query_masks=None):
    """Compute retrieval metrics from a similarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing captions from the dataset

    Returns:
        (dict[str:float]): retrieval metrics

    NOTES: We find the closest "GT caption" in the style of VSE, which corresponds
    to finding the rank of the closest relevant caption in embedding space:
    github.com/ryankiros/visual-semantic-embedding/blob/master/evaluation.py#L52-L56
    """
    # switch axes of text and video
    sims = sims.T

    if False:
        # experiment with toy example
        sims = np.ones((3, 3))
        sims[0, 0] = 2
        sims[1, 1:2] = 2
        sims[2, :] = 2
        query_masks = None

    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_caps = sims.shape
    dists = -sims
    caps_per_video = num_caps // num_queries
    break_ties = "averaging"

    MISSING_VAL = 1E8
    query_ranks = []
    for ii in range(num_queries):
        row_dists = dists[ii, :]
        if query_masks is not None:
            # Set missing queries to have a distance of infinity.  A missing query
            # refers to a query position `n` for a video that had less than `n`
            # captions (for example, a few MSRVTT videos only have 19 queries)
            row_dists[np.logical_not(query_masks.reshape(-1))] = MISSING_VAL

        # NOTE: Using distance subtraction to perform the ranking is easier to make
        # deterministic than using argsort, which suffers from the issue of defining
        # "stability" for equal distances.  Example of distance subtraction code:
        # github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/train.py
        sorted_dists = np.sort(row_dists)

        min_rank = np.inf
        for jj in range(ii * caps_per_video, (ii + 1) * caps_per_video):
            if row_dists[jj] == MISSING_VAL:
                # skip rankings of missing captions
                continue
            ranks = np.where((sorted_dists - row_dists[jj]) == 0)[0]
            if break_ties == "optimistically":
                rank = ranks[0]
            elif break_ties == "averaging":
                # NOTE: If there is more than one caption per video, its possible for the
                # method to do "worse than chance" in the degenerate case when all
                # similarities are tied.  TODO(Samuel): Address this case.
                rank = ranks.mean()
            if rank < min_rank:
                min_rank = rank
        query_ranks.append(min_rank)
    query_ranks = np.array(query_ranks)

    # sanity check against old version of code
    if False:
        sorted_dists = np.sort(dists, axis=1)
        gt_dists_old = np.diag(dists)
        gt_dists_old = gt_dists_old[:, np.newaxis]
        rows_old, cols_old = np.where((sorted_dists - gt_dists_old) == 0)
        if rows_old.size > num_queries:
            _, idx = np.unique(rows_old, return_index=True)
            cols_old = cols_old[idx]
        num_diffs = (1 - (cols_old == query_ranks)).sum()
        msg = f"new metric doesn't match in {num_diffs} places"
        assert np.array_equal(cols_old, query_ranks), msg

        # visualise the distance matrix
        import sys
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
        from zsvision.zs_iterm import zs_dispFig  # NOQA
        plt.matshow(dists)
        zs_dispFig()

    return cols2metrics(query_ranks, num_queries)


def retrieval_as_classification(sims, query_masks=None):
    """Compute classification metrics from a similarity matrix.
    """
    assert sims.ndim == 2, "expected a matrix"

    # switch axes of query-labels and video
    sims = sims.T
    query_masks = query_masks.T
    dists = -sims
    num_queries, num_labels = sims.shape
    break_ties = "averaging"

    query_ranks = []
    for ii in range(num_queries):
        row_dists = dists[ii, :]

        # NOTE: Using distance subtraction to perform the ranking is easier to make
        # deterministic than using argsort, which suffers from the issue of defining
        # "stability" for equal distances.  Example of distance subtraction code:
        # github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/train.py
        sorted_dists = np.sort(row_dists)

        # min_rank = np.inf
        label_ranks = []
        for gt_label in np.where(query_masks[ii, :])[0]:
            ranks = np.where((sorted_dists - row_dists[gt_label]) == 0)[0]
            if break_ties == "optimistically":
                rank = ranks[0]
            elif break_ties == "averaging":
                # NOTE: If there is more than one caption per video, its possible for the
                # method to do "worse than chance" in the degenerate case when all
                # similarities are tied.  TODO(Samuel): Address this case.
                rank = ranks.mean()
            else:
                raise ValueError(f"unknown tie-breaking method: {break_ties}")
            label_ranks.append(rank)
        # Avoid penalising for assigning higher similarity to other gt labels. This is
        # done by subtracting out the better ranked query labels.  Note that this step
        # introduces a slight skew in favour of videos with lots of labels.  We can
        # address this later with a normalisation step if needed.
        label_ranks = [x - idx for idx, x in enumerate(label_ranks)]

        # Include all labels in the final calculation
        query_ranks.extend(label_ranks)
    query_ranks = np.array(query_ranks)

    # sanity check against old version of code
    if False:
        # visualise the distance matrix
        import sys
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
        from zsvision.zs_iterm import zs_dispFig  # NOQA
        # plt.matshow(dists)
        # zs_dispFig()
        plt.hist(query_ranks, bins=313, alpha=0.5)
        plt.grid()
        zs_dispFig()
        import ipdb;
        ipdb.set_trace()

    return cols2metrics(query_ranks, num_queries=len(query_ranks))


def cols2metrics(cols, num_queries):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics


def mean_average_precision(sims, query_masks=None):
    ap_meter = APMeter()
    ap_meter.add(output=sims.T, target=query_masks.T)
    return {"mAP": ap_meter.value().mean()}

def acc(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def video_precision(output, target):
    """ percentage of videos which have been aligned to a matching text pair"""
    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    correct = 0
    for bout, btarg in zip(output, target):
        for pair in bout:
            eq = torch.eq(pair, btarg)
            if torch.logical_and(eq[:, 0], eq[:, 1]).any():
                correct += 1
    return correct / (target.shape[0] * target.shape[1])

def video_precision_adj(output, target):
    """ adjusts the video precision metric by ignoring videos which have no aligning text."""
    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    correct = 0
    for bout, btarg in zip(output, target):
        for pair in bout:
            eq = torch.eq(pair, btarg)
            if torch.logical_and(eq[:, 0], eq[:, 1]).any():
                correct += 1
    denom = len(target[:, :, 0].unique())

    return correct / denom

def video_precision_adj(output, target):
    """ adjusts the video precision metric by ignoring videos which have no aligning text."""
    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    assert output.shape[0] == target.shape[0]
    assert output.shape[2] == target.shape[2] == 2

    correct = 0
    for bout, btarg in zip(output, target):
        for pair in bout:
            eq = torch.eq(pair, btarg)
            if torch.logical_and(eq[:, 0], eq[:, 1]).any():
                correct += 1
    denom = len(target[:, :, 0].unique())

    return correct / denom

def ntu_t2v_metrics(sims, action_labels, query_masks=None):
    """NTU 데이터셋을 위한 Text-to-Video 검색 메트릭
    
    Args:
        sims (np.array): [num_unique_labels, num_videos] 유사도 매트릭스
        action_labels (list): 각 비디오의 액션 라벨 (길이: num_videos)
        query_masks: 사용하지 않음 (NTU에서는 필요 없음)
    
    Returns:
        dict: 검색 성능 메트릭
    """
    assert sims.ndim == 2, "expected a matrix"
    num_queries, num_vids = sims.shape  # [60, 2880]
    
    # 액션 라벨들을 유니크하게 정렬
    unique_labels = sorted(list(set(action_labels)))
    assert len(unique_labels) == num_queries, f"쿼리 수({num_queries})와 유니크 라벨 수({len(unique_labels)})가 맞지 않음"
    
    # 각 라벨별로 해당하는 비디오 인덱스들 구하기
    label_to_video_indices = {}
    for video_idx, label in enumerate(action_labels):
        if label not in label_to_video_indices:
            label_to_video_indices[label] = []
        label_to_video_indices[label].append(video_idx)
    
    query_ranks = []
    
    for query_idx, query_label in enumerate(unique_labels):
        # 현재 쿼리 라벨에 해당하는 비디오들의 인덱스
        gt_video_indices = label_to_video_indices[query_label]
        
        # 현재 쿼리에 대한 모든 비디오의 유사도 점수
        query_sims = sims[query_idx, :]  # shape: [num_videos]
        
        # 유사도를 거리로 변환 (높은 유사도 = 낮은 거리)
        query_dists = -query_sims
        
        # 모든 비디오를 거리 순으로 정렬
        sorted_indices = np.argsort(query_dists)
        
        # 정답 비디오들의 랭크 찾기 (첫 번째 정답의 랭크 사용)
        min_rank = float('inf')
        for gt_idx in gt_video_indices:
            rank = np.where(sorted_indices == gt_idx)[0][0] + 1  # 1-based rank
            min_rank = min(min_rank, rank)
        
        query_ranks.append(min_rank)
    
    query_ranks = np.array(query_ranks)
    
    # 메트릭 계산
    r1 = 100 * np.mean(query_ranks <= 1)
    r5 = 100 * np.mean(query_ranks <= 5)
    r10 = 100 * np.mean(query_ranks <= 10)
    r50 = 100 * np.mean(query_ranks <= 50)
    medr = np.median(query_ranks)
    meanr = np.mean(query_ranks)
    
    return {
        "R1": r1,
        "R5": r5, 
        "R10": r10,
        "R50": r50,
        "MedR": medr,
        "MeanR": meanr,
        "query_ranks": query_ranks
    }


def ntu_v2t_metrics(sims, action_labels, query_masks=None):
    """NTU 데이터셋을 위한 Video-to-Text 검색 메트릭
    
    Args:
        sims (np.array): [num_unique_labels, num_videos] 유사도 매트릭스  
        action_labels (list): 각 비디오의 액션 라벨 (길이: num_videos)
        query_masks: 사용하지 않음
        
    Returns:
        dict: 검색 성능 메트릭
    """
    # sims를 transpose: [num_videos, num_unique_labels]
    sims = sims.T
    
    assert sims.ndim == 2, "expected a matrix"
    num_videos, num_labels = sims.shape  # [2880, 60]
    
    # 유니크 라벨들 정렬
    unique_labels = sorted(list(set(action_labels)))
    assert len(unique_labels) == num_labels, f"라벨 수({num_labels})와 유니크 라벨 수({len(unique_labels)})가 맞지 않음"
    
    query_ranks = []
    
    for video_idx in range(num_videos):
        # 현재 비디오의 정답 라벨
        gt_label = action_labels[video_idx]
        gt_label_idx = unique_labels.index(gt_label)
        
        # 현재 비디오에 대한 모든 라벨의 유사도 점수
        video_sims = sims[video_idx, :]  # shape: [num_labels]
        
        # 유사도를 거리로 변환
        video_dists = -video_sims
        
        # 모든 라벨을 거리 순으로 정렬
        sorted_indices = np.argsort(video_dists)
        
        # 정답 라벨의 랭크 찾기
        rank = np.where(sorted_indices == gt_label_idx)[0][0] + 1  # 1-based rank
        query_ranks.append(rank)
    
    query_ranks = np.array(query_ranks)
    
    # 메트릭 계산
    r1 = 100 * np.mean(query_ranks <= 1)
    r5 = 100 * np.mean(query_ranks <= 5)
    r10 = 100 * np.mean(query_ranks <= 10)
    r50 = 100 * np.mean(query_ranks <= 50)
    medr = np.median(query_ranks)
    meanr = np.mean(query_ranks)
    
    return {
        "R1": r1,
        "R5": r5,
        "R10": r10, 
        "R50": r50,
        "MedR": medr,
        "MeanR": meanr,
        "query_ranks": query_ranks
    }
