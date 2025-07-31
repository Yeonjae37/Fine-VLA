import os
import numpy as np
from datetime import datetime


def save_ntu_t2v_results(sims, action_labels, unique_labels, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    label_to_video_indices = {}
    for video_idx, label in enumerate(action_labels):
        if label not in label_to_video_indices:
            label_to_video_indices[label] = []
        label_to_video_indices[label].append(video_idx)
    
    results = []
    query_ranks = []
    
    for query_idx, query_label in enumerate(unique_labels):
        gt_video_indices = set(label_to_video_indices[query_label])
        
        query_sims = sims[query_idx, :]
        
        sorted_indices = np.argsort(-query_sims)
        
        top_k = min(10, len(sorted_indices))
        query_results = []
        
        first_correct_rank = None
        correct_count = 0
        
        for rank, video_idx in enumerate(sorted_indices[:top_k]):
            is_correct = video_idx in gt_video_indices
            if is_correct:
                correct_count += 1
                if first_correct_rank is None:
                    first_correct_rank = rank + 1
            
            query_results.append({
                'rank': rank + 1,
                'video_idx': video_idx,
                'video_label': action_labels[video_idx],
                'similarity': float(query_sims[video_idx]),
                'is_correct': is_correct,
                'correct_symbol': '✓' if is_correct else '✗'
            })

        if first_correct_rank is None:
            first_correct_rank = len(sorted_indices) + 1
        query_ranks.append(first_correct_rank)
        
        results.append({
            'query_idx': query_idx,
            'query_label': query_label,
            'gt_video_count': len(gt_video_indices),
            'first_correct_rank': first_correct_rank,
            'top50_correct_count': correct_count,
            'results': query_results
        })

    query_ranks = np.array(query_ranks)
    r1 = 100 * np.mean(query_ranks <= 1)
    r5 = 100 * np.mean(query_ranks <= 5)
    r10 = 100 * np.mean(query_ranks <= 10)
    r50 = 100 * np.mean(query_ranks <= 50)
    medr = np.median(query_ranks)
    meanr = np.mean(query_ranks)

    filename = os.path.join(save_dir, f"t2v_results_{timestamp}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NTU RGB+D Dataset - Text-to-Video Retrieval Results\n")
        f.write("=" * 80 + "\n")

        f.write("-" * 40 + "\n")
        f.write(f"R@1:  {r1:6.2f}%\n")
        f.write(f"R@5:  {r5:6.2f}%\n")
        f.write(f"R@10: {r10:6.2f}%\n")
        f.write(f"MedR: {medr:6.1f}\n")
        f.write(f"MeanR: {meanr:6.1f}\n\n")

        for result in results:
            f.write(f"Query: '{result['query_label']}' (Label {result['query_idx']+1}/{len(unique_labels)})\n")
            f.write(f"gt video count: {result['gt_video_count']}개\n")
            
            f.write("Rank  Video_Idx  Video_Label           Similarity  Correct\n")
            
            for res in result['results']:
                f.write(f"{res['rank']:4d}  {res['video_idx']:9d}  {res['video_label']:20s}  "
                       f"{res['similarity']:8.4f}  {res['correct_symbol']:>7s}\n")
            f.write("\n")

    return filename


def save_ntu_v2t_results(sims, action_labels, unique_labels, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sims_t = sims.T
    num_videos, num_labels = sims_t.shape
    
    results = []
    query_ranks = []

    sample_size = min(100, num_videos)
    sample_indices = np.linspace(0, num_videos-1, sample_size, dtype=int)
    
    for sample_idx, video_idx in enumerate(sample_indices):
        gt_label = action_labels[video_idx]
        gt_label_idx = unique_labels.index(gt_label)
        
        video_sims = sims_t[video_idx, :]
        
        sorted_indices = np.argsort(-video_sims)
        
        rank = np.where(sorted_indices == gt_label_idx)[0][0] + 1
        query_ranks.append(rank)
       
        top_k = min(20, len(sorted_indices))
        video_results = []
        
        for r, label_idx in enumerate(sorted_indices[:top_k]):
            is_correct = (label_idx == gt_label_idx)
            video_results.append({
                'rank': r + 1,
                'label_idx': label_idx,
                'label': unique_labels[label_idx],
                'similarity': float(video_sims[label_idx]),
                'is_correct': is_correct,
                'correct_symbol': '✓' if is_correct else '✗'
            })
        
        results.append({
            'video_idx': video_idx,
            'gt_label': gt_label,
            'gt_rank': rank,
            'results': video_results
        })
    
    all_ranks = []
    for video_idx in range(num_videos):
        gt_label = action_labels[video_idx]
        gt_label_idx = unique_labels.index(gt_label)
        video_sims = sims_t[video_idx, :]
        sorted_indices = np.argsort(-video_sims)
        rank = np.where(sorted_indices == gt_label_idx)[0][0] + 1
        all_ranks.append(rank)
    
    all_ranks = np.array(all_ranks)
    r1 = 100 * np.mean(all_ranks <= 1)
    r5 = 100 * np.mean(all_ranks <= 5)
    r10 = 100 * np.mean(all_ranks <= 10)
    r50 = 100 * np.mean(all_ranks <= 50)
    medr = np.median(all_ranks)
    meanr = np.mean(all_ranks)
 
    filename = os.path.join(save_dir, f"v2t_results_{timestamp}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NTU RGB+D Dataset - Video-to-Text Retrieval Results\n")
        f.write("=" * 80 + "\n")

        f.write("-" * 40 + "\n")
        f.write(f"R@1:  {r1:6.2f}%\n")
        f.write(f"R@5:  {r5:6.2f}%\n")
        f.write(f"R@10: {r10:6.2f}%\n")
        f.write(f"R@50: {r50:6.2f}%\n")
        f.write(f"MedR: {medr:6.1f}\n")
        f.write(f"MeanR: {meanr:6.1f}\n\n")

        for result in results:
            f.write(f"Video idx {result['video_idx']} (gt label: '{result['gt_label']}')\n")
            f.write("Rank  Label_Idx  Label                 Similarity  Correct\n")
            for res in result['results']:
                f.write(f"{res['rank']:4d}  {res['label_idx']:9d}  {res['label']:20s}  "
                       f"{res['similarity']:8.4f}  {res['correct_symbol']:>7s}\n")
            f.write("\n")

    return filename


def save_ntu_results(sims, action_labels, unique_labels, save_dir="results"):
    t2v_file = save_ntu_t2v_results(sims, action_labels, unique_labels, save_dir)
    v2t_file = save_ntu_v2t_results(sims, action_labels, unique_labels, save_dir)
    return t2v_file, v2t_file