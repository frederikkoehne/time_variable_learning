import json
import torch
from torch import nn

from tau_modules import tau_modules


class Experiment:
    """Base class for tau learning experiment
    """

    def __init__(self):
        pass

    def update_trackers(self):
        self.losses.append(float(self.loss))
        self.tau_tracking.append([float(tau) for tau in self.taus])
        self.clean_losses.append(float(self.clean_loss))

    def evaluate_loss(self):
        self.loss = self.loss_fn(self.model(self.Yfull), self.Cfull)
        return self.loss

    def zero_grad(self):
        return self.model.zero_grad()

    @torch.no_grad()
    def perform_training_step(self):
        """Performs classic (S)GD training step.
        """
        for p in self.model.parameters():
            p.add_(p.grad, alpha=-self.lr)

    def save_trackers_to_json(self, path: str):
        result = {
            key: getattr(self, key) for key in self.trackers
        }
        result['lr'] = self.lr
        result['n_epochs'] = self.n_epochs
        if hasattr(self, 'meta'):
            result.update(self.meta)
        with open(path, 'w') as file:
            json.dump(result, file)

    def run_experiment(self):
        raise NotImplementedError()

    @property
    def trainable_taus(self):
        return self.taus

    def tau_blocks(self):
        for module in self.model.children():
            if isinstance(module, tau_modules.TauBlock):
                yield module

    @torch.no_grad()
    def prune_network(self, crit=.01):
        taus = self.trainable_taus_list

        tau_abs_sum = sum(abs(tau) for tau in taus)
        fractions = [abs(tau) / tau_abs_sum for tau in taus]

        if min(fractions) < crit:  # Test if the network is going to be pruned
            print('Pruning!')

            crit_tau = min(taus)

            new_model_modules = []
            for child in self.model.children():
                if isinstance(child, tau_modules.TauBlock) and child.tau is crit_tau:
                    continue

                new_model_modules.append(child)

            self.model = nn.Sequential(*new_model_modules)

            # Update taus:
            self.taus = []
            self.trainable_taus_list = []
            for child in self.model.children():
                if isinstance(child, tau_modules.TauBlock):
                    self.taus.append(child.tau)
                    if child.tau.requires_grad:
                        self.trainable_taus_list.append(child.tau)

        # Re-init optimizer
        self.init_optimizer(self.optimizer_class, lr=self.lr)

    def init_data(self, data, n_epochs=None, batch_size=None, train=True):
        if train:
            assert batch_size is not None and n_epochs is not None, 'Batch size and Number of Epochs needed when initializing train data'
            self.batch_size = batch_size
            self.n_epochs = n_epochs

            self.data = data
            self.dataloader = torch.utils.data.DataLoader(
                data, batch_size=batch_size, shuffle=True)
            self.has_data = True
            self.meta['batch_size'] = batch_size
            self.meta['n_epochs'] = n_epochs
            return

        self.test_data = data
        self.has_test_data = True

    @torch.no_grad()
    def evaluate_on_test_set(self, model=None):
        assert self.has_test_data, 'No Test Data given. Set Test data with init_data method'
        if model is None:
            model = self.model

        batch_size = len(self.test_data)
        while batch_size > 0:

            test_loss = 0.
            percentage_right = 0.
            total = 0
            try:
                dataloader = torch.utils.data.DataLoader(
                    self.test_data, batch_size=batch_size)
                for x, y in dataloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = model(x)
                    loss, _ = self.loss_fn(pred, y)
                    test_loss += loss * len(x)
                    number_right = sum(pred.argmax(-1) == y)
                    percentage_right += number_right
                    total += len(x)
                percentage_right /= total
                test_loss /= total
                break
            except RuntimeError as e:
                print(e)
                batch_size = batch_size // 2
        return {
            'loss': test_loss,
            'acc': percentage_right
        }

    @property
    def trainable_taus(self):
        return self.trainable_taus_list

    def evaluate_loss(self, x, target):
        self.loss, self.clean_loss = self.loss_fn(self.model(x), target)
        return self.loss, self.clean_loss

    def reset_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.data, batch_size=self.batch_size, shuffle=True)

    def init_optimizer(self, optimizer_class, **kwargs):
        self.optimizer = optimizer_class(
            params=self.model.parameters(), **kwargs)
        self.lr = kwargs['lr']
        self.optimizer_class = optimizer_class

    def run_experiment(self, train_log_interval=100, evaluate_on_test_set=True):
        if evaluate_on_test_set:
            eval_res = self.evaluate_on_test_set()
            self.test_losses.append(float(eval_res['loss']))
            self.test_accs.append(float(eval_res['acc']))

        assert self.has_data, 'No Dataset given at the moment. Use general_experiment.init_data to initialize'
        for e in range(self.n_epochs):
            for k, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                loss, clean_loss = self.evaluate_loss(x, y)
                loss.backward()
                self.optimizer.step()

                # last tau projection step:

                if self.dependent_last_tau:
                    with torch.no_grad():
                        self.dependent_tau.copy_(
                            self.T - sum(self.trainable_taus_list))

                self.optimizer.zero_grad()
                self.update_trackers()
                if k % train_log_interval == 0:
                    print(loss)

            if evaluate_on_test_set:
                eval_res = self.evaluate_on_test_set()
                self.test_losses.append(float(eval_res['loss']))
                self.test_accs.append(float(eval_res['acc']))

            if self.adaptive_pruning:

                self.prune_network()


class DenseResnetExperiment(Experiment):
    """Class for a dense ResNet tau learning experiment"""

    def __init__(self,
                 in_dimension,
                 out_dimension,
                 loss_fn,
                 preprocessing_layer=None,
                 internal_dimension=100,
                 number_of_blocks=3,
                 activation=None,
                 T=1.,
                 tau_l2_factor=None,
                 tau_l2_reg=None,
                 l1_reg_for_taus=None,
                 device=None,
                 tau_is_trainable=True,
                 use_tau_for_first_layer=False,
                 adaptive_pruning=False,
                 dependent_last_tau=False
                 ) -> None:
        """
        Initializes a dense ResNet tau learning experiment.

        Args:
            in_dimension (int): Input dimension
            out_dimension (int): Output dimension
            loss_fn: Loss function
            preprocessing_layer: Preprocessing layer (default: None)
            internal_dimension (int): Internal dimension (default: 100)
            number_of_blocks (int): Number of blocks (default: 3)
            activation: Activation function. If None, ReLU is used (default: None)
            T (float): Initial tau value (default: 1.0)
            tau_l2_factor: Tau L2 factor (default: None)
            tau_l2_reg: Tau L2 regularization (default: None)
            l1_reg_for_taus: L1 regularization for taus (default: None)
            device: Device to run the experiment on. If None, cuda is used, if available (default: None)
            tau_is_trainable (bool): Whether taus are trainable (default: True)
            use_tau_for_first_layer (bool): Whether to use tau for the first layer (default: False)
            adaptive_pruning (bool): Whether to perform adaptive pruning (default: False)
            dependent_last_tau (bool): Whether the last tau is determined from T and the other taus by enforcing the sum over all taus to be equal T (default: False)
        """

        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        number_of_blocks_helper = number_of_blocks
        if use_tau_for_first_layer:
            number_of_blocks_helper += 1

        self.dependent_last_tau = dependent_last_tau

        self.dependent_tau = None
        self.T = T

        layers = []
        tau_blocks = []
        self.taus = []
        self.trainable_taus_list = []

        if preprocessing_layer is not None:
            layers.append(preprocessing_layer)

        if use_tau_for_first_layer:
            first_tau_block = tau_modules.TauBlock(
                nn.Linear(in_dimension, internal_dimension),
                add_shortcut=in_dimension == out_dimension,
                activation=activation,
                initial_tau=T/number_of_blocks_helper,
                tau_is_trainable=tau_is_trainable
            )
            tau_blocks.append(first_tau_block)
            self.taus.append(first_tau_block.tau)
            self.trainable_taus_list.append(first_tau_block.tau)

        else:
            layers.append(nn.Linear(in_dimension, internal_dimension))
            layers.append(activation)

        for k in range(number_of_blocks):

            this_tau_is_trainable = not (
                dependent_last_tau and k == number_of_blocks - 1)

            tau_blocks.append(
                tau_modules.TauBlock(
                    nn.Linear(internal_dimension, internal_dimension),
                    activation=activation,
                    initial_tau=T/number_of_blocks_helper,
                    tau_is_trainable=this_tau_is_trainable
                )
            )

            self.taus.append(tau_blocks[-1].tau)
            if this_tau_is_trainable:
                self.trainable_taus_list.append(tau_blocks[-1].tau)
            else:
                self.dependent_tau = tau_blocks[-1].tau

        for layer in tau_blocks:
            layers.append(layer)
        layers.append(nn.Linear(internal_dimension, out_dimension, bias=False))

        self.model = nn.Sequential(*layers).to(self.device)

        self.has_data = False
        self.has_test_data = False

        def _loss_fn(pred, target):
            res = loss_fn(pred, target)
            reg_term = 0.
            if tau_l2_reg:
                reg_term += tau_l2_reg * sum(tau ** 2 for tau in self.taus)

            if tau_l2_factor:
                reg_term += tau_l2_factor * (sum(self.taus) - T) ** 2
            if l1_reg_for_taus:
                reg_term += l1_reg_for_taus * sum(abs(tau) for tau in self.taus)
            return res + reg_term, res

        self.loss_fn = _loss_fn

        self.test_losses = []
        self.test_accs = []
        self.losses = []
        self.clean_losses = []
        self.tau_tracking = []
        self.trackers = ['losses', 'tau_tracking',
                         'test_losses', 'test_accs', 'clean_losses']
        self.adaptive_pruning = adaptive_pruning

        self.meta = {
            'tau_is_trainable': tau_is_trainable,
            'l1_reg': 0. if l1_reg_for_taus is None else l1_reg_for_taus,
            'l2_reg': 0. if tau_l2_reg is None else tau_l2_reg,
            'tau_sum_factor': 0. if tau_l2_factor is None else tau_l2_factor,
            'T': T,
            'internal_dimension': internal_dimension,
            'number_of_tau_blocks': number_of_blocks,
            'adaptive_pruning': adaptive_pruning,
            'dependent_last_tau': dependent_last_tau
        }

    def run_experiment(self, train_log_interval=100, evaluate_on_test_set=True):

        if evaluate_on_test_set:
            eval_res = self.evaluate_on_test_set()
            self.test_losses.append(float(eval_res['loss']))
            self.test_accs.append(float(eval_res['acc']))

        assert self.has_data, 'No Dataset given at the moment. Use init_data to initilize'
        for e in range(self.n_epochs):
            for k, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                loss, clean_loss = self.evaluate_loss(x, y)
                loss.backward()
                self.optimizer.step()

                # last tau projection step:

                if self.dependent_last_tau:
                    with torch.no_grad():
                        self.dependent_tau.copy_(
                            self.T - sum(self.trainable_taus_list))

                self.optimizer.zero_grad()
                self.update_trackers()
                if k % train_log_interval == 0:
                    print(loss, len(self.trainable_taus_list))

            if evaluate_on_test_set:
                eval_res = self.evaluate_on_test_set()
                self.test_losses.append(float(eval_res['loss']))
                self.test_accs.append(float(eval_res['acc']))
            if self.adaptive_pruning:

                self.prune_network()


class DenseFracDNNExperiment(Experiment):
    """Class for a dense FracDNN tau learning experiment"""

    def __init__(self,
                 in_dimension,
                 out_dimension,
                 loss_fn,
                 preprocessing_layer=None,
                 internal_dimension=100,
                 number_of_blocks=3,
                 activation=None,
                 T=1.,
                 tau_l2_factor=None,
                 tau_l2_reg=None,
                 l1_reg_for_taus=None,
                 device=None,
                 tau_is_trainable=True,
                 dependent_last_tau=False
                 ):
        """
        Initializes a dense FracDNN tau learning experiment.

        Args:
            in_dimension (int): Input dimension
            out_dimension (int): Output dimension
            loss_fn: Loss function
            preprocessing_layer: Preprocessing layer (default: None)
            internal_dimension (int): Internal dimension (default: 100)
            number_of_blocks (int): Number of blocks (default: 3)
            activation: Activation function. If None, ReLU is used (default: None)
            T (float): Initial tau value (default: 1.0)
            tau_l2_factor: Tau L2 factor (default: None)
            tau_l2_reg: Tau L2 regularization (default: None)
            l1_reg_for_taus: L1 regularization for taus (default: None)
            device: Device to run the experiment on. If None, cuda is used, if available (default: None)
            tau_is_trainable (bool): Whether taus are trainable (default: True)
            dependent_last_tau (bool): Whether the last tau is determined from T and the other taus by enforcing the sum over all taus to be equal T (default: False)
        """
        super().__init__()

        # to be consistent with resnets, as here tau is used for the first layer anytime.
        number_of_blocks += 1

        if activation is None:
            activation = nn.ReLU()
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.adaptive_pruning = False

        self.dependent_last_tau = dependent_last_tau

        self.dependent_tau = None
        self.T = T

        self.tau_is_trainable = [tau_is_trainable] * number_of_blocks

        if self.dependent_last_tau:
            self.tau_is_trainable[-1] = False

        layers = []
        self.taus = []
        self.trainable_taus_list = []

        if preprocessing_layer is not None:
            layers.append(preprocessing_layer)

        fracdnn_layers = []
        fracdnn_activations = []

        fracdnn_layers.append(nn.Linear(in_dimension, internal_dimension))
        fracdnn_activations.append(activation)
        for k in range(number_of_blocks-1):
            fracdnn_layers.append(
                nn.Linear(internal_dimension, internal_dimension))
            fracdnn_activations.append(activation)

        self.frac_dnn = tau_modules.FractionalDNN(
            layers=fracdnn_layers,
            activations=fracdnn_activations,
            initial_taus=[T/number_of_blocks] * number_of_blocks,
            tau_is_trainable=tau_is_trainable,
            eps=1e-4,
            project_taus_before_forward_pass=True
        )

        self.taus = self.frac_dnn.taus

        if dependent_last_tau:
            self.trainable_taus_list = self.taus[:-1]
            self.dependent_tau = self.taus[-1]
        else:
            self.trainable_taus_list = self.taus

        self.model = nn.Sequential(
            preprocessing_layer,
            self.frac_dnn.to(self.device),
            nn.Linear(internal_dimension, out_dimension, bias=False)
        ).to(self.device)

        self.has_data = False
        self.has_test_data = False

        def _loss_fn(pred, target):
            res = loss_fn(pred, target)
            reg_term = 0.
            if tau_l2_reg:
                reg_term += tau_l2_reg * sum(tau ** 2 for tau in self.taus)

            if tau_l2_factor:
                reg_term += tau_l2_factor * (sum(self.taus) - T) ** 2
            if l1_reg_for_taus:
                reg_term += l1_reg_for_taus * sum(abs(tau) for tau in self.taus)
            return res + reg_term, res

        self.loss_fn = _loss_fn

        self.test_losses = []
        self.test_accs = []
        self.losses = []
        self.clean_losses = []
        self.tau_tracking = []
        self.trackers = ['losses', 'tau_tracking',
                         'test_losses', 'test_accs', 'clean_losses']

        self.meta = {
            'tau_is_trainable': tau_is_trainable,
            'l1_reg': 0. if l1_reg_for_taus is None else l1_reg_for_taus,
            'l2_reg': 0. if tau_l2_reg is None else tau_l2_reg,
            'tau_sum_factor': 0. if tau_l2_factor is None else tau_l2_factor,
            'T': T,
            'internal_dimension': internal_dimension,
            'number_of_tau_blocks': number_of_blocks,
            'adaptive_pruning': self.adaptive_pruning,
            'dependent_last_tau': dependent_last_tau
        }
