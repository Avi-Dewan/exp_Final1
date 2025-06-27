# METHOD 1: EXPLICIT CLUSTER CENTER SEPARATION
# =============================================================================




def METHOD1_PI_CL_softBCE_train(model, 
                                labeled_train_loader,
                                labeled_eval_loader,
                                unlabeled_train_loader,
                                unlabeled_eval_loader,
                                args):
    """
    METHOD 1: Explicit Cluster Center Separation
    Following the exact flow of CE_PI_CL_softBCE_train but with separated cluster centers
    
    Training with:
    - Cross-entropy loss for labeled data
    - KL on sharpened targets for unlabeled data
    - MSE consistency loss
    - SimCLR contrastive loss
    - Pairwise BCE loss (ranking-based)
    - Separated cluster centers with separation loss
    """
    # Replace the clustering mechanism with separated cluster head
    separated_cluster_head = SeparatedClusterHead(
        feat_dim=args.proj_dim_unlabeled,  # 20
        n_labeled_classes=args.n_labeled_classes,
        n_unlabeled_classes=args.n_unlabeled_classes
    ).to(device)
    
    # Initialize unlabeled centers with the original centers from init_prob_kmeans
    separated_cluster_head.unlabeled_centers.data = model.encoder.center.data.clone()
    
    simCLR_loss = SimCLR_Loss(batch_size=args.batch_size, temperature=0.5).to(device)
    criterion_bce = softBCE_N()
    ce_criterion = nn.CrossEntropyLoss()

    # Include separated cluster head parameters in optimizer
    all_params = list(model.parameters()) + list(separated_cluster_head.parameters())
    optimizer = SGD(all_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    accuracies, nmi_scores, ari_scores, f1_scores = [], [], [], []

    for epoch in range(args.epochs):
        model.train()
        separated_cluster_head.train()
        loss_record = AverageMeter()

        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)

        labeled_loader_iter = iter(labeled_train_loader)
        
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(unlabeled_train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)

            extracted_feat, _, z_unlabeled = model(x)
            extracted_feat_bar, _, z_unlabeled_bar = model(x_bar)

            # Use separated cluster head for unlabeled data
            prob = separated_cluster_head(z_unlabeled, is_labeled=False)
            prob_bar = separated_cluster_head(z_unlabeled_bar, is_labeled=False)

            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)

            # contrastive loss using internal projector
            z_i = model.projector_CL(extracted_feat)
            z_j = model.projector_CL(extracted_feat_bar)
            contrastive_loss = simCLR_loss(z_i, z_j)

            # pairwise BCE label via ranking
            rank_feat = extracted_feat.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            matches = (rank_idx1.unsqueeze(2) == rank_idx2.unsqueeze(1)).sum(dim=2)
            common_elements = matches.sum(dim=1)
            pairwise_pseudo_label = common_elements.float() / args.topk

            prob_pair, _ = PairEnum(prob)
            _, prob_bar_pair = PairEnum(prob_bar)

            bce_loss = criterion_bce(prob_pair, prob_bar_pair, pairwise_pseudo_label)

            # === Add labeled data processing ===
            try:
                x_l, y_l, _ = next(labeled_loader_iter)
            except StopIteration:
                labeled_loader_iter = iter(labeled_train_loader)
                x_l, y_l, _ = next(labeled_loader_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            _, labeled_pred_l, z_l = model(x_l)

            # Standard CE loss for labeled classification
            ce_loss = ce_criterion(labeled_pred_l, y_l)
            
            # NEW: Labeled cluster assignment using separated cluster head
            prob_l = separated_cluster_head(z_l, is_labeled=True)
            labeled_cluster_loss = F.cross_entropy(prob_l, y_l)
            
            # NEW: Separation loss between labeled and unlabeled centers
            sep_loss = separation_loss(
                separated_cluster_head.labeled_centers,
                separated_cluster_head.unlabeled_centers,
                margin=2.0
            )

            # Total loss (following your original flow + new components)
            loss = (ce_loss + labeled_cluster_loss + sharp_loss + 
                   w * consistency_loss + w * contrastive_loss + 
                   w_softBCE * bce_loss + 0.5 * sep_loss)
            
            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Avg Loss = {loss_record.avg:.4f}")
        
        # Update model.encoder.center for compatibility with existing test function
        model.encoder.center.data = separated_cluster_head.unlabeled_centers.data.clone()
        
        acc, nmi, ari, probs = test(model, unlabeled_eval_loader, args)
        f1 = test_labeled(model, labeled_eval_loader)
        
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)
        f1_scores.append(f1)

        if epoch % args.update_interval == 0:
            print("Updating p_targets...")
            args.p_targets = target_distribution(probs)

    # Save model (including separated cluster head)
    torch.save({
        'state_dict': model.state_dict(), 
        'center': model.encoder.center,
        'separated_cluster_head': separated_cluster_head.state_dict()
    }, args.model_dir)
    print(f"Model saved to {args.model_dir}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), accuracies, label="ACC")
    plt.plot(range(args.epochs), nmi_scores, label="NMI")
    plt.plot(range(args.epochs), ari_scores, label="ARI")
    plt.plot(range(args.epochs), f1_scores, label="F1(labeled data)")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("METHOD 1: Training Metrics with Separated Cluster Centers")
    plt.legend()
    plt.savefig(args.model_folder + '/method1_accuracies.png')


def PI_CL_softBCE_unlabeled_only_train(model, 
                                               unlabeled_train_loader,
                                               unlabeled_eval_loader,
                                               args):
    """
    METHOD 1 variant: Unlabeled-only version following PI_CL_softBCE_train flow
    This assumes you have some way to get labeled centers (e.g., from pre-training)
    """
    # For unlabeled-only, we still create separated head but only use unlabeled part
    separated_cluster_head = SeparatedClusterHead(
        feat_dim=args.proj_dim_unlabeled,  # 20
        n_labeled_classes=args.n_labeled_classes,
        n_unlabeled_classes=args.n_unlabeled_classes
    ).to(device)
    
    # Initialize unlabeled centers with the original centers from init_prob_kmeans
    separated_cluster_head.unlabeled_centers.data = model.encoder.center.data.clone()
    
    # Initialize labeled centers with some reference (e.g., from pre-trained model or fixed values)
    # This could be loaded from the pre-trained model or set to fixed values
    # For now, using Xavier initialization as in your original code

    separated_cluster_head.labeled_centers.data = model.encoder.labeledCenter.data.clone()
    
    simCLR_loss = SimCLR_Loss(batch_size=args.batch_size, temperature=0.5).to(device)
    criterion_bce = softBCE_N()

    # Include separated cluster head parameters in optimizer
    all_params = list(model.parameters()) + list(separated_cluster_head.parameters())
    optimizer = SGD(all_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    accuracies, nmi_scores, ari_scores = [], [], []

    for epoch in range(args.epochs):
        model.train()
        separated_cluster_head.train()
        loss_record = AverageMeter()

        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)

        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(unlabeled_train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)

            extracted_feat, labeled_pred, z_unlabeled = model(x)
            extracted_feat_bar, labeled_pred_bar, z_unlabeled_bar = model(x_bar)

            # Use separated cluster head for unlabeled data
            prob = separated_cluster_head(z_unlabeled, is_labeled=False)
            prob_bar = separated_cluster_head(z_unlabeled_bar, is_labeled=False)

            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)

            # contrastive loss using internal projector
            z_i = model.projector_CL(extracted_feat)
            z_j = model.projector_CL(extracted_feat_bar)
            contrastive_loss = simCLR_loss(z_i, z_j)

            # pairwise BCE label via ranking
            rank_feat = extracted_feat.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            matches = (rank_idx1.unsqueeze(2) == rank_idx2.unsqueeze(1)).sum(dim=2)
            common_elements = matches.sum(dim=1)
            pairwise_pseudo_label = common_elements.float() / args.topk

            prob_pair, _ = PairEnum(prob)
            _, prob_bar_pair = PairEnum(prob_bar)

            bce_loss = criterion_bce(prob_pair, prob_bar_pair, pairwise_pseudo_label)
            
            # NEW: Separation loss between labeled and unlabeled centers
            sep_loss = separation_loss(
                separated_cluster_head.labeled_centers,
                separated_cluster_head.unlabeled_centers,
                margin=2.0
            )

            # Total loss (following your original PI_CL_softBCE flow + separation)
            loss = sharp_loss + w * consistency_loss + w * contrastive_loss + w_softBCE * bce_loss + 0.5 * sep_loss
            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Avg Loss = {loss_record.avg:.4f}")
        
        # Update model.encoder.center for compatibility with existing test function
        model.encoder.center.data = separated_cluster_head.unlabeled_centers.data.clone()
        
        acc, nmi, ari, probs = test(model, unlabeled_eval_loader, args)
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)

        if epoch % args.update_interval == 0:
            print("Updating p_targets...")
            args.p_targets = target_distribution(probs)

    # Save model
    torch.save({
        'state_dict': model.state_dict(), 
        'center': model.encoder.center,
        'separated_cluster_head': separated_cluster_head.state_dict()
    }, args.model_dir)
    print(f"Model saved to {args.model_dir}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), accuracies, label="Accuracy")
    plt.plot(range(args.epochs), nmi_scores, label="NMI")
    plt.plot(range(args.epochs), ari_scores, label="ARI")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("METHOD 1: Training Metrics (Unlabeled Only)")
    plt.legend()
    plt.savefig(args.model_folder + '/accuracies.png')     


# =============================================================================
# METHOD 3: CONTRASTIVE FEATURE SPACE SEPARATION
# =============================================================================

def contrastive_separation_loss(labeled_features, unlabeled_features, temperature=0.5):
    """
    Contrastive loss to separate labeled and unlabeled feature spaces
    """
    # Normalize features
    labeled_norm = F.normalize(labeled_features, dim=1)
    unlabeled_norm = F.normalize(unlabeled_features, dim=1)
    
    # Compute similarities
    sim_matrix = torch.mm(labeled_norm, unlabeled_norm.t()) / temperature
    
    # We want to minimize these similarities (push apart)
    # Use negative log likelihood of pushing apart
    separation_loss = torch.logsumexp(sim_matrix, dim=1).mean()
    
    return separation_loss

def METHOD3_PI_CL_softBCE_contrastive_train(model, 
                                           labeled_train_loader,
                                           labeled_eval_loader,
                                           unlabeled_train_loader, 
                                           unlabeled_eval_loader, 
                                           args):
    """
    METHOD 3: Contrastive Feature Space Separation following PI_CL_softBCE flow
    - KL on sharpened targets
    - MSE consistency loss
    - SimCLR contrastive loss
    - Pairwise BCE loss (ranking-based)
    - Cross-entropy loss for labeled data
    - Contrastive separation: push labeled and unlabeled feature spaces apart
    """
    simCLR_loss = SimCLR_Loss(batch_size=args.batch_size, temperature=0.5).to(device)
    criterion_bce = softBCE_N()
    ce_criterion = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    accuracies, nmi_scores, ari_scores, f1_scores = [], [], [], []

    for epoch in range(args.epochs):
        model.train()
        loss_record = AverageMeter()

        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)
        w_contrastive_sep = 0.2 * ramps.sigmoid_rampup(epoch, args.rampup_length)  # Gradual contrastive separation weight

        labeled_loader_iter = iter(labeled_train_loader)
        
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(unlabeled_train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)

            extracted_feat, labeled_pred, z_unlabeled = model(x)
            extracted_feat_bar, labeled_pred_bar, z_unlabeled_bar = model(x_bar)

            prob = feat2prob(z_unlabeled, model.encoder.center)
            prob_bar = feat2prob(z_unlabeled_bar, model.encoder.center)

            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)

            # contrastive loss using internal projector
            z_i = model.projector_CL(extracted_feat)
            z_j = model.projector_CL(extracted_feat_bar)
            contrastive_loss = simCLR_loss(z_i, z_j)

            # pairwise BCE label via ranking
            rank_feat = extracted_feat.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            matches = (rank_idx1.unsqueeze(2) == rank_idx2.unsqueeze(1)).sum(dim=2)
            common_elements = matches.sum(dim=1)
            pairwise_pseudo_label = common_elements.float() / args.topk

            prob_pair, _ = PairEnum(prob)
            _, prob_bar_pair = PairEnum(prob_bar)

            bce_loss = criterion_bce(prob_pair, prob_bar_pair, pairwise_pseudo_label)

            # === Add labeled CE loss ===
            try:
                x_l, y_l, _ = next(labeled_loader_iter)
            except StopIteration:
                labeled_loader_iter = iter(labeled_train_loader)
                x_l, y_l, _ = next(labeled_loader_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            extracted_feat_l, labeled_pred_l, z_l = model(x_l)

            ce_loss = ce_criterion(labeled_pred_l, y_l)

            # === Add contrastive feature space separation ===
            # Push labeled and unlabeled feature spaces apart
            contrastive_sep_loss = contrastive_separation_loss(extracted_feat_l, extracted_feat, temperature=0.5)
            
            # Total loss (same as PI_CL_softBCE but with CE and contrastive separation)
            loss = ce_loss + sharp_loss + w * consistency_loss + w * contrastive_loss + w_softBCE * bce_loss + w_contrastive_sep * contrastive_sep_loss
            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Avg Loss = {loss_record.avg:.4f}")
        acc, nmi, ari, probs = test(model, unlabeled_eval_loader, args)
        f1 = test_labeled(model, labeled_eval_loader)
        
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)
        f1_scores.append(f1)

        if epoch % args.update_interval == 0:
            print("Updating p_targets...")
            args.p_targets = target_distribution(probs)

    # Save model
    torch.save({
        'state_dict': model.state_dict(), 
        'center': model.encoder.center,
        'labeled_center': model.encoder.labeledCenter
    }, args.model_dir)
    print(f"Model saved to {args.model_dir}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), accuracies, label="ACC")
    plt.plot(range(args.epochs), nmi_scores, label="NMI")
    plt.plot(range(args.epochs), ari_scores, label="ARI")
    plt.plot(range(args.epochs), f1_scores, label="F1(labeled data)")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("METHOD 3: Contrastive Feature Separation Training Metrics")
    plt.legend()
    plt.savefig(args.model_folder + '/accuracies.png')


# =============================================================================
# METHOD 4: ADAPTIVE MARGIN BOUNDARY LEARNING
# =============================================================================

def adaptive_margin_loss(unlabeled_features, labeled_centers, unlabeled_centers, margin_factor=1.0):
    """
    Adaptive margin loss to maintain optimal distance between labeled and unlabeled clusters
    """
    # Distances from unlabeled features to labeled centers
    dist_to_labeled = torch.cdist(unlabeled_features, labeled_centers)  # [batch, n_labeled]
    min_dist_to_labeled = dist_to_labeled.min(dim=1)[0]  # [batch]
    
    # Distances from unlabeled features to unlabeled centers
    dist_to_unlabeled = torch.cdist(unlabeled_features, unlabeled_centers)  # [batch, n_unlabeled]
    min_dist_to_unlabeled = dist_to_unlabeled.min(dim=1)[0]  # [batch]
    
    # Adaptive margin: closer to unlabeled centers, farther from labeled centers
    # margin = max(0, margin_factor - (min_dist_to_labeled - min_dist_to_unlabeled))
    margin_violation = torch.relu(margin_factor - (min_dist_to_labeled - min_dist_to_unlabeled))
    
    return margin_violation.mean()

def intra_cluster_cohesion_loss(features, centers, assignments):
    """
    Encourage samples to be close to their assigned cluster centers
    """
    batch_size = features.size(0)
    n_clusters = centers.size(0)
    
    # Get soft assignments
    distances = torch.cdist(features, centers)  # [batch, n_clusters]
    soft_assignments = F.softmax(-distances, dim=1)  # [batch, n_clusters]
    
    # Weighted distance to centers
    weighted_distances = (distances * soft_assignments).sum(dim=1)  # [batch]
    
    return weighted_distances.mean()

def METHOD4_PI_CL_softBCE_adaptive_margin_train(model, 
                                               labeled_train_loader,
                                               labeled_eval_loader,
                                               unlabeled_train_loader, 
                                               unlabeled_eval_loader, 
                                               args):
    """
    METHOD 4: Adaptive Margin Boundary Learning following PI_CL_softBCE flow
    - KL on sharpened targets
    - MSE consistency loss
    - SimCLR contrastive loss
    - Pairwise BCE loss (ranking-based)
    - Cross-entropy loss for labeled data
    - Adaptive margin loss: maintain optimal boundaries between labeled/unlabeled spaces
    - Intra-cluster cohesion: encourage tight clusters
    """
    simCLR_loss = SimCLR_Loss(batch_size=args.batch_size, temperature=0.5).to(device)
    criterion_bce = softBCE_N()
    ce_criterion = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    accuracies, nmi_scores, ari_scores, f1_scores = [], [], [], []

    for epoch in range(args.epochs):
        model.train()
        loss_record = AverageMeter()

        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        w_softBCE = args.rampup_coefficient_softBCE * ramps.sigmoid_rampup(epoch, args.rampup_length_softBCE)
        w_margin = 0.5 * ramps.sigmoid_rampup(epoch, args.rampup_length)  # Gradual margin weight
        w_cohesion = 0.3 * ramps.sigmoid_rampup(epoch, args.rampup_length)  # Cohesion weight

        labeled_loader_iter = iter(labeled_train_loader)
        
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(unlabeled_train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)

            extracted_feat, labeled_pred, z_unlabeled = model(x)
            extracted_feat_bar, labeled_pred_bar, z_unlabeled_bar = model(x_bar)

            prob = feat2prob(z_unlabeled, model.encoder.center)
            prob_bar = feat2prob(z_unlabeled_bar, model.encoder.center)

            sharp_loss = F.kl_div(prob.log(), args.p_targets[idx].float().to(device))
            consistency_loss = F.mse_loss(prob, prob_bar)

            # contrastive loss using internal projector
            z_i = model.projector_CL(extracted_feat)
            z_j = model.projector_CL(extracted_feat_bar)
            contrastive_loss = simCLR_loss(z_i, z_j)

            # pairwise BCE label via ranking
            rank_feat = extracted_feat.detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            matches = (rank_idx1.unsqueeze(2) == rank_idx2.unsqueeze(1)).sum(dim=2)
            common_elements = matches.sum(dim=1)
            pairwise_pseudo_label = common_elements.float() / args.topk

            prob_pair, _ = PairEnum(prob)
            _, prob_bar_pair = PairEnum(prob_bar)

            bce_loss = criterion_bce(prob_pair, prob_bar_pair, pairwise_pseudo_label)

            # === Add labeled CE loss ===
            try:
                x_l, y_l, _ = next(labeled_loader_iter)
            except StopIteration:
                labeled_loader_iter = iter(labeled_train_loader)
                x_l, y_l, _ = next(labeled_loader_iter)

            x_l, y_l = x_l.to(device), y_l.to(device)
            _, labeled_pred_l, z_l = model(x_l)

            ce_loss = ce_criterion(labeled_pred_l, y_l)

            # === Add adaptive margin boundary learning ===
            # Maintain optimal boundaries between labeled and unlabeled spaces
            margin_loss = adaptive_margin_loss(
                z_unlabeled, 
                model.encoder.labeledCenter, 
                model.encoder.center, 
                margin_factor=2.0
            )
            
            # Encourage intra-cluster cohesion for unlabeled data
            cohesion_loss = intra_cluster_cohesion_loss(
                z_unlabeled, 
                model.encoder.center, 
                prob
            )
            
            # Total loss (same as PI_CL_softBCE but with CE, margin, and cohesion)
            loss = ce_loss + sharp_loss + w * consistency_loss + w * contrastive_loss + w_softBCE * bce_loss + w_margin * margin_loss + w_cohesion * cohesion_loss
            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Avg Loss = {loss_record.avg:.4f}")
        acc, nmi, ari, probs = test(model, unlabeled_eval_loader, args)
        f1 = test_labeled(model, labeled_eval_loader)
        
        accuracies.append(acc)
        nmi_scores.append(nmi)
        ari_scores.append(ari)
        f1_scores.append(f1)

        if epoch % args.update_interval == 0:
            print("Updating p_targets...")
            args.p_targets = target_distribution(probs)

    # Save model
    torch.save({
        'state_dict': model.state_dict(), 
        'center': model.encoder.center,
        'labeled_center': model.encoder.labeledCenter
    }, args.model_dir)
    print(f"Model saved to {args.model_dir}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(args.epochs), accuracies, label="ACC")
    plt.plot(range(args.epochs), nmi_scores, label="NMI")
    plt.plot(range(args.epochs), ari_scores, label="ARI")
    plt.plot(range(args.epochs), f1_scores, label="F1(labeled data)")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Score")
    plt.title("METHOD 4: Adaptive Margin Boundary Learning Training Metrics")
    plt.legend()
    plt.savefig(args.model_folder + '/accuracies.png')
