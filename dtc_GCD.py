


def PI_CL_softBCE_knowledge_preserving_train(model, labeled_loader, unlabeled_loader, eval_loader, args):
    """
    Knowledge-preserving training that:
    1. Maintains performance on labeled classes
    2. Discovers novel classes without collapse
    3. Uses labeled data as anchors/guidance
    """
    
    simCLR_loss = SimCLR_Loss(batch_size=args.batch_size, temperature=0.5).to(device)
    projector = ProjectionHead(512 * BasicBlock.expansion, 2048, 128).to(device)
    criterion_bce = softBCE_N()
    criterion_ce = nn.CrossEntropyLoss()  # For labeled data
    
    # Separate optimizers for different components
    optimizer_backbone = SGD(model.parameters(), lr=args.lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay)  # Lower LR for backbone
    optimizer_projector = SGD(projector.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Store original labeled class centers for regularization
    labeled_centers = model.center[:args.n_labeled_classes].clone().detach()
    
    accuracies = []
    nmi_scores = []
    ari_scores = []
    labeled_accuracies = []  # Track labeled class performance
    
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        labeled_loss_record = AverageMeter()
        novel_loss_record = AverageMeter()
        
        model.train()
        
        # Ramp-up schedules
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)
        w_separation = args.separation_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)  # New: separation loss weight
        
        # Combined training on both labeled and unlabeled data
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        max_iters = max(len(labeled_loader), len(unlabeled_loader))
        
        for batch_idx in range(max_iters):
            total_loss = 0
            
            # === LABELED DATA PROCESSING ===
            try:
                (x_labeled, x_labeled_aug), labels_true, labeled_idx = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                (x_labeled, x_labeled_aug), labels_true, labeled_idx = next(labeled_iter)
            
            x_labeled = x_labeled.to(device)
            x_labeled_aug = x_labeled_aug.to(device)
            labels_true = labels_true.to(device)
            
            # Forward pass on labeled data
            extracted_feat_labeled, final_feat_labeled = model(x_labeled)
            extracted_feat_labeled_aug, final_feat_labeled_aug = model(x_labeled_aug)
            
            # Supervised loss on labeled data
            supervised_loss = criterion_ce(final_feat_labeled, labels_true)
            
            # Consistency loss on labeled augmentations
            prob_labeled = feat2prob(final_feat_labeled, model.center)
            prob_labeled_aug = feat2prob(final_feat_labeled_aug, model.center)
            labeled_consistency_loss = F.mse_loss(prob_labeled, prob_labeled_aug)
            
            # Contrastive loss on labeled data
            z_labeled, z_labeled_aug = projector(extracted_feat_labeled), projector(extracted_feat_labeled_aug)
            labeled_contrastive_loss = simCLR_loss(z_labeled, z_labeled_aug)
            
            labeled_total_loss = supervised_loss + 0.5 * labeled_consistency_loss + 0.5 * labeled_contrastive_loss
            labeled_loss_record.update(labeled_total_loss.item(), x_labeled.size(0))
            
            # === UNLABELED DATA PROCESSING ===
            try:
                (x_unlabeled, x_unlabeled_aug), _, unlabeled_idx = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (x_unlabeled, x_unlabeled_aug), _, unlabeled_idx = next(unlabeled_iter)
            
            x_unlabeled = x_unlabeled.to(device)
            x_unlabeled_aug = x_unlabeled_aug.to(device)
            
            # Forward pass on unlabeled data
            extracted_feat_unlabeled, final_feat_unlabeled = model(x_unlabeled)
            extracted_feat_unlabeled_aug, final_feat_unlabeled_aug = model(x_unlabeled_aug)
            
            # Probability distributions for unlabeled data
            prob_unlabeled = feat2prob(final_feat_unlabeled, model.center)
            prob_unlabeled_aug = feat2prob(final_feat_unlabeled_aug, model.center)
            
            # Sharp loss (only on novel classes)
            novel_class_mask = torch.arange(args.n_labeled_classes, args.n_labeled_classes + args.n_unlabeled_classes).to(device)
            prob_unlabeled_novel = prob_unlabeled[:, novel_class_mask]
            prob_unlabeled_novel = F.softmax(prob_unlabeled_novel, dim=1)
            
            # Adjust target distribution to focus on novel classes
            unlabeled_targets = args.p_targets[unlabeled_idx].float().to(device)
            unlabeled_targets_novel = unlabeled_targets[:, novel_class_mask]
            unlabeled_targets_novel = F.softmax(unlabeled_targets_novel, dim=1)
            
            sharp_loss = F.kl_div(prob_unlabeled_novel.log(), unlabeled_targets_novel)
            
            # Consistency loss on unlabeled data
            consistency_loss = F.mse_loss(prob_unlabeled, prob_unlabeled_aug)
            
            # Contrastive loss on unlabeled data
            z_unlabeled, z_unlabeled_aug = projector(extracted_feat_unlabeled), projector(extracted_feat_unlabeled_aug)
            unlabeled_contrastive_loss = simCLR_loss(z_unlabeled, z_unlabeled_aug)
            
            # === SEPARATION LOSS ===
            # Ensure novel classes don't collapse to labeled classes
            prob_labeled_classes = prob_unlabeled[:, :args.n_labeled_classes]  # Prob of being labeled classes
            prob_novel_classes = prob_unlabeled[:, args.n_labeled_classes:]     # Prob of being novel classes
            
            # Encourage unlabeled data to have low probability for labeled classes
            separation_loss = torch.mean(prob_labeled_classes)  # Minimize prob of labeled classes
            
            # === PAIRWISE BCE LOSS (Modified) ===
            # Only apply to novel class relationships
            rank_feat = extracted_feat_unlabeled.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            
            matches = (rank_idx1.unsqueeze(2) == rank_idx2.unsqueeze(1)).sum(dim=2)
            common_elements = matches.sum(dim=1)
            pairwise_pseudo_label = common_elements.float() / args.topk
            
            prob_pair, _ = PairEnum(prob_unlabeled_novel)  # Only novel classes
            _, prob_aug_pair = PairEnum(prob_unlabeled_aug[:, novel_class_mask])
            
            bce_loss = criterion_bce(prob_pair, prob_aug_pair, pairwise_pseudo_label)
            
            # === CENTER REGULARIZATION ===
            # Prevent labeled class centers from drifting
            center_drift_loss = F.mse_loss(model.center[:args.n_labeled_classes], labeled_centers)
            
            # Combine losses
            novel_total_loss = (sharp_loss + 
                              w * consistency_loss + 
                              w * unlabeled_contrastive_loss + 
                              w_softBCE * bce_loss + 
                              w_separation * separation_loss + 
                              0.1 * center_drift_loss)
            
            novel_loss_record.update(novel_total_loss.item(), x_unlabeled.size(0))
            
            # Total loss
            total_loss = labeled_total_loss + novel_total_loss
            loss_record.update(total_loss.item(), x_labeled.size(0) + x_unlabeled.size(0))
            
            # Backward pass
            optimizer_backbone.zero_grad()
            optimizer_projector.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            
            optimizer_backbone.step()
            optimizer_projector.step()
        
        print(f'Epoch {epoch}: Total Loss: {loss_record.avg:.4f}, '
              f'Labeled Loss: {labeled_loss_record.avg:.4f}, '
              f'Novel Loss: {novel_loss_record.avg:.4f}')
        
        # Evaluation
        acc, nmi, ari, probs = test_novel_classes(model, eval_loader, args)
        labeled_acc = test_labeled_classes(model, labeled_loader, args)
        
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)
        labeled_accuracies.append(labeled_acc)
        
        print(f'Novel Classes - ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}')
        print(f'Labeled Classes - ACC: {labeled_acc:.4f}')
        
        # Update target distribution
        if epoch % args.update_interval == 0:
            print('Updating target distribution...')
            # Only update for novel classes
            novel_probs = probs[:, args.n_labeled_classes:]
            args.p_targets[:, args.n_labeled_classes:] = target_distribution(novel_probs)
    
    # Save model
    model_dict = {'state_dict': model.state_dict(), 'center': model.center}
    torch.save(model_dict, args.model_dir)
    
    # Plot results
    plot_dual_metrics(accuracies, nmi_scores, ari_scores, labeled_accuracies, args)
    
    return model


def test_labeled_classes(model, labeled_loader, args):
    """Test performance on labeled classes only"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for (x, _), labels, _ in labeled_loader:
            x, labels = x.to(device), labels.to(device)
            _, features = model(x)
            
            # Get predictions for labeled classes only
            labeled_logits = features[:, :args.n_labeled_classes]
            _, predicted = torch.max(labeled_logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total


def test_novel_classes(model, eval_loader, args):
    """Test performance on novel classes only"""
    model.eval()
    preds = np.array([])
    targets = np.array([])
    probs = np.zeros((len(eval_loader.dataset), model.center.size(0)))
    
    with torch.no_grad():
        for batch_idx, (x, label, idx) in enumerate(tqdm(eval_loader)):
            x, label = x.to(device), label.to(device)
            _, features = model(x)
            
            # Get probabilities for all classes
            prob = feat2prob(features, model.center)
            
            # For novel class evaluation, only consider novel class predictions
            novel_prob = prob[:, args.n_labeled_classes:]
            _, pred = novel_prob.max(1)
            pred = pred + args.n_labeled_classes  # Adjust indices
            
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            
            idx = idx.data.cpu().numpy()
            probs[idx, :] = prob.cpu().detach().numpy()
    
    # Adjust targets for novel class evaluation
    targets = targets - args.n_labeled_classes
    preds = preds - args.n_labeled_classes
    
    acc = cluster_acc(targets.astype(int), preds.astype(int))
    nmi = nmi_score(targets, preds)
    ari = ari_score(targets, preds)
    
    return acc, nmi, ari, torch.from_numpy(probs)


def plot_dual_metrics(novel_acc, nmi_scores, ari_scores, labeled_acc, args):
    """Plot both novel and labeled class performance"""
    plt.figure(figsize=(15, 5))
    
    epochs_range = range(len(novel_acc))
    
    # Novel class metrics
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, novel_acc, 'b-', label="Novel ACC")
    plt.plot(epochs_range, nmi_scores, 'r-', label="NMI")
    plt.plot(epochs_range, ari_scores, 'g-', label="ARI")
    plt.xlabel("Epochs")
    plt.ylabel("Novel Class Metrics")
    plt.title("Novel Class Discovery Performance")
    plt.legend()
    
    # Labeled class performance
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, labeled_acc, 'orange', label="Labeled ACC")
    plt.axhline(y=0.95, color='red', linestyle='--', label="Target (95%)")
    plt.xlabel("Epochs")
    plt.ylabel("Labeled Class Accuracy")
    plt.title("Labeled Class Retention")
    plt.legend()
    
    # Combined view
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, novel_acc, 'b-', label="Novel ACC")
    plt.plot(epochs_range, labeled_acc, 'orange', label="Labeled ACC")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Overall Performance")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(args.model_folder + '/comprehensive_metrics.png')
    plt.close()