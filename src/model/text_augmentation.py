import torch


def get_text_augmentation_templates():
    text_aug = [
        "a video of action {}", 
        "a picture of action {}", 
        "Human action of {}", 
        "{}, an action",
        "{} this is an action", 
        "{}, a video of action", 
        "Playing action of {}", 
        "{}",
        "Playing a kind of action, {}", 
        "Doing a kind of action, {}", 
        "Look, the human is {}",
        "Can you recognize the action of {}?", 
        "Video classification of {}", 
        "A video of {}",
        "The man is {}", 
        "The woman is {}"
    ]
    return text_aug


def augment_text_labels(unique_labels):
    templates = get_text_augmentation_templates()
    
    augmented_texts = []
    label_groups = []
    
    for label_idx, label in enumerate(unique_labels):
        group_indices = []
        
        for template in templates:
            augmented_text = template.format(label)
            augmented_texts.append(augmented_text)
            group_indices.append(len(augmented_texts) - 1)
        
        label_groups.append(group_indices)
    
    return {
        'augmented_texts': augmented_texts,
        'label_groups': label_groups
    }


def average_augmented_embeddings(text_embeds, label_groups):
    num_labels = len(label_groups)
    embed_dim = text_embeds.shape[1]
    
    averaged_embeds = torch.zeros(num_labels, embed_dim, device=text_embeds.device)
    
    for label_idx, group_indices in enumerate(label_groups):
        label_embeds = text_embeds[group_indices]  # [num_templates, embed_dim]
        averaged_embeds[label_idx] = label_embeds.mean(dim=0)
    
    return averaged_embeds