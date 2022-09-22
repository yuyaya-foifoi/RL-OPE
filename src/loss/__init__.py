from src.loss.kld_bce import KLD_BCE


def get_loss_function(name: str):
    if name == "BCE":
        return KLD_BCE
