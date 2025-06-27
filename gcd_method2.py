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


