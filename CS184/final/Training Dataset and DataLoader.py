import time
from comfig import *
from Model import model
from training import dl_train


if __name__ == '__main__':
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    n_batches = len(dl_train)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Starting epoch {epoch} of {NUM_EPOCHS}")

        time_start = time.time()
        loss_accum = 0.0
        loss_mask_accum = 0.0
        for batch_idx, (images, targets) in enumerate(dl_train, 1):
            print(batch_idx)
            # Predict
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            loss_mask = loss_dict['loss_mask'].item()
            loss_accum += loss.item()
            loss_mask_accum += loss_mask
            print(batch_idx)
            if batch_idx % 50 == 0:
                print(
                    f"    [Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}. Mask-only loss: {loss_mask:7.3f}")

            if USE_SCHEDULER:
                lr_scheduler.step()

        # Train losses
            train_loss = loss_accum / n_batches
            train_loss_mask = loss_mask_accum / n_batches

            elapsed = time.time() - time_start

            torch.save(model.state_dict(), f"pytorch_model-e{epoch}.bin")
            prefix = f"[Epoch {epoch:2d} / {NUM_EPOCHS:2d}]"
            print(f"{prefix} Train mask-only loss: {train_loss_mask:7.3f}")
            print(f"{prefix} Train loss: {train_loss:7.3f}. [{elapsed:.0f} secs]")

