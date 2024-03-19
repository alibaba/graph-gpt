import torch

# from torcheval.metrics import BinaryAUROC, BinaryAccuracy
from torcheval import metrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, Accuracy

# metric from `torchmetrics` will collect and update across all GPUs, while `torcheval.metrics` NOT!!!
from . import control_flow

_metrics = control_flow.Register()
get_metrics = _metrics.build  # return func results
get_metrics_cls = _metrics.get  # return the func


@_metrics("single_label_classification")
class SingleLabelClassificationMetrics:
    def __init__(self, device, num_labels=2, **kwargs):
        self.device = device
        self.num_labels = num_labels
        if num_labels == 2:
            self.auroc_metric = BinaryAUROC().to(device)
            self.acc_metric = BinaryAccuracy().to(device)
        else:
            self.auroc_metric = None
            self.acc_metric = Accuracy(task="multiclass", num_classes=num_labels).to(
                device
            )
        self.auroc = None
        self.acc = None
        self.all_pred = None
        self.all_labels = None
        self.all_idx = None
        self.ls_pred = []
        self.ls_labels = []
        self.ls_idx = []

    def update(self, logits, labels, idx):
        if self.num_labels == 2:
            y_pred = logits.softmax(axis=-1)[:, 1].to(self.device)
            self.auroc_metric.update(y_pred, labels)
        else:
            y_pred = torch.argmax(logits, dim=-1)  # [bz, vocab] -> [bz]
        self.acc_metric.update(y_pred, labels)

        # cpu_device = torch.device("cpu")
        if self.num_labels == 2:
            y_pred = logits[:, 1].float() - logits[:, 0].float()  # .to(cpu_device)
        else:
            y_pred = y_pred  # .to(cpu_device)

        self.ls_pred.append(y_pred)
        self.ls_labels.append(labels)  # .to(cpu_device))
        self.ls_idx.append(idx)  # .to(cpu_device))

    def compute(self):
        self.auroc = (
            self.auroc_metric.compute().item() if self.auroc_metric is not None else -1
        )
        self.acc = self.acc_metric.compute().item()
        self._finalize()

    def _finalize(self):
        self.all_pred = torch.hstack(self.ls_pred)  # [N_samples]
        self.all_labels = torch.hstack(self.ls_labels)  # [N_samples]
        self.all_idx = torch.hstack(self.ls_idx)  # [N_samples]

    def to_dict(self):
        self._finalize()
        return {"y_true": self.all_labels, "y_pred": self.all_pred, "idx": self.all_idx}

    def get_output_shape(self, dim, key=None):
        return dim

    def results_in_tuple(self):
        return self.auroc, self.acc

    def results_in_str_tuple(self):
        return str(self.auroc), str(self.acc)

    def results_in_details(self, prefix=""):
        return f"{prefix} AUROC: {self.auroc}, {prefix} ACC: {self.acc}"

    def results_in_dict(self, prefix=""):
        return {f"{prefix} ACC": self.acc}  # f"{prefix} AUROC": {self.auroc},


@_metrics("multi_label_classification")
class MultiLabelClassificationMetrics:
    def __init__(self, device, num_labels=2, **kwargs):
        self.device = device
        self.num_labels = num_labels
        self.auroc_metric = metrics.BinaryAUROC(num_tasks=num_labels).to(device)
        self.auroc_vec = None
        self.auroc_mean = None
        self.ls_logits = []
        self.ls_labels = []
        self.ls_idx = []

    def update(self, logits, labels, idx):
        y_pred = logits.float().sigmoid().T.to(self.device)  # [num_labels, batch]
        y_true = labels.T  # [num_labels, batch]
        self.auroc_metric.update(y_pred, y_true)

        self.ls_logits.append(logits.float())  # [batch, num_labels]
        self.ls_labels.append(labels)  # [batch, num_labels]
        self.ls_idx.append(idx)  # [batch]

    def compute(self):
        self.auroc_vec = self.auroc_metric.compute().cpu().numpy()  # [num_labels]
        self.auroc_mean = self.auroc_vec.mean()  # scalar

    def to_dict(self):
        # y_pred = torch.cat(self.auroc_metric.inputs, -1).T  # [num_nodes, num_labels]
        # y_true = torch.cat(self.auroc_metric.targets, -1).T  # [num_nodes, num_labels]
        y_pred = torch.vstack(self.ls_logits)  # [num_nodes/num_graphs, num_labels]
        y_true = torch.vstack(self.ls_labels)  # [num_nodes/num_graphs, num_labels]
        idx = torch.cat(self.ls_idx)  # [num_nodes/num_graphs]
        return {"y_true": y_true, "y_pred": y_pred, "idx": idx}

    def get_output_shape(self, dim, key=None):
        if key == "idx":
            return dim
        else:
            return dim, self.num_labels

    def results_in_tuple(self):
        return [self.auroc_mean]

    def results_in_str_tuple(self):
        return [str(self.auroc_mean)]

    def results_in_details(self, prefix=""):
        return f"{prefix} mean AUROC: {self.auroc_mean}"

    def results_in_full_details(self, prefix=""):
        return f"{prefix} mean AUROC: {self.auroc_mean}, detailed AUROC: {','.join(self.auroc_vec.astype(str))}"


@_metrics("regression")
class RegressionMetrics:
    def __init__(self, device, num_labels=1, **kwargs):
        self.device = device
        if num_labels == 1:
            self.mse_metric = MeanSquaredError().to(device)
            self.mae_metric = MeanAbsoluteError().to(device)
            self.mse = None
            self.mae = None
            self.ls_logits = []
            self.ls_labels = []
            self.ls_idx = []
        else:
            raise NotImplementedError(
                f"Metrics for regression with num_labels={num_labels} is not implemented yet!"
            )

    def update(self, logits, labels, idx):
        y_pred = logits.reshape(-1).to(self.device)
        self.mse_metric.update(y_pred, labels)
        self.mae_metric.update(y_pred, labels)

        self.ls_logits.append(logits.reshape(-1))  # [batch]
        self.ls_labels.append(labels.reshape(-1))  # [batch]
        self.ls_idx.append(idx)  # [batch]

    def compute(self):
        self.mse = self.mse_metric.compute().item()
        self.mae = self.mae_metric.compute().item()

    def to_dict(self):
        y_pred = torch.cat(self.ls_logits)  # [num_nodes/num_graphs]
        y_true = torch.cat(self.ls_labels)  # [num_nodes/num_graphs]
        idx = torch.cat(self.ls_idx)  # [num_nodes/num_graphs]
        return {"y_true": y_true, "y_pred": y_pred, "idx": idx}

    def get_output_shape(self, dim, key=None):
        return dim

    def results_in_tuple(self):
        return self.mse, self.mae

    def results_in_str_tuple(self):
        return str(self.mse), str(self.mae)

    def results_in_details(self, prefix=""):
        return f"{prefix} MSE: {self.mse}, {prefix} MAE: {self.mae}"
