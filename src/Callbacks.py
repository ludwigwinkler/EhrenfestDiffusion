import lightning
import pyrootutils


def get_callbacks(cfg):
    callbacks = []
    path = pyrootutils.find_root(search_from=__file__, indicator=[".git"])
    path = path / f"checkpoints/{cfg.logging.run_id}"
    path.mkdir(parents=True, exist_ok=True)
    ckpt = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath=path,
        filename=f"{cfg.logging.run_id}" + '-{epoch:02d}',
        save_top_k=1,
        mode="min",
        monitor="Validation/Loss",
        every_n_train_steps=10_000,
        save_last=True,
    )
    callbacks += [ckpt]
    lr = lightning.pytorch.callbacks.LearningRateMonitor()
    callbacks += [lr]

    return callbacks
