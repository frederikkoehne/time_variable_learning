import torch
import itertools

import utils


       
def main_frac_dnn():
    """Performs all experiments for the FracDNN Architecture.
    """
    number_of_runs = 1
    number_of_tau_blocks = 5
    internal_dimension = 100
    n_epochs = 200
    batch_size = 100
    lr = 1e-2
    
    options = {
        'use_fmnist': [True, False],
        'tau_is_trainable': [True, False],
        'l1_reg_for_taus': [None, 0.01],
        'tau_l2_factor': [None, 0.5],
        'T': [1, number_of_tau_blocks + 1],
        'dependent_last_tau': [True, False]
    }
    
    keys = options.keys()
    value_combinations = itertools.product(*options.values())
    
    
    for seed in range(number_of_runs):
        for values in value_combinations:
            
            curr_setting = dict(zip(keys, values))
            if not is_of_interest(curr_setting):
                continue

            torch.manual_seed(seed)
            experiment = utils.mnist_frac_dnn_exp_wrapper(
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=1e-2,
                **curr_setting,
                number_of_blocks=number_of_tau_blocks,
                internal_dimension=internal_dimension
            )
            
            print('Starting experiment with setting:', curr_setting)
            experiment.run_experiment()
            
            
            name = 'results/frac_dnn/'
            name += f'{"fmnist" if curr_setting["use_fmnist"] else "mnist"}/'
            
            utils.check_and_create_directory(name)
            
            name += ('tau_trainable' if curr_setting['tau_is_trainable'] else 'tau_not_trainable') + '__'
            name += f'l1_reg_for_taus_{curr_setting["l1_reg_for_taus"]}__'
            name += f'tau_l2_factor_{curr_setting["tau_l2_factor"]}__'
            name += f'T_{curr_setting["T"]}__'
            name += f'dependent_tau_{curr_setting["dependent_last_tau"]}'
            
            name += f'__seed_{seed}'
            
            name += '.json'
            
            print('Saving to', name)
            experiment.save_trackers_to_json(name)

def main_res_net():
    """Performs all experiments for the ResNet Architecture.
    """
    number_of_runs = 1
    
    number_of_tau_blocks = 5
    internal_dimension = 100
    n_epochs = 200
    batch_size = 100
    lr = 1e-2
    
    options = {
            'use_fmnist': [True, False],
            'tau_is_trainable': [True, False],
            'l1_reg_for_taus': [None, 0.01],
            'tau_l2_factor': [None, 0.5],
            'adaptive_pruning': [True, False],
            'T': [1, number_of_tau_blocks + 1],
            'dependent_last_tau': [True, False]
    }
    

    keys = options.keys()
    value_combinations = itertools.product(*options.values())
    
    
    for seed in range(number_of_runs):
        for values in value_combinations:
            
            curr_setting = dict(zip(keys, values))
            if not is_of_interest(curr_setting):
                continue

            torch.manual_seed(seed)
            experiment = utils.mnist_exp_wrapper(
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                **curr_setting,
                number_of_blocks=number_of_tau_blocks,
                internal_dimension=internal_dimension
            )
            
            print('Starting experiment with setting:', curr_setting)
            experiment.run_experiment()
            
            name = 'results/resnet/'
            name += f'{"fmnist" if curr_setting["use_fmnist"] else "mnist"}/'
            
            utils.check_and_create_directory(name)
            
            name += ('tau_trainable' if curr_setting['tau_is_trainable'] else 'tau_not_trainable') + '__'
            name += f'l1_reg_for_taus_{curr_setting["l1_reg_for_taus"]}__'
            name += f'tau_l2_factor_{curr_setting["tau_l2_factor"]}__'
            name += f'adaptive_pruning_{curr_setting["adaptive_pruning"]}__'
            name += f'T_{curr_setting["T"]}__'
            name += f'dependent_tau_{curr_setting["dependent_last_tau"]}'
            
            name += f'__seed_{seed}'
            
            name += '.json'
            
            print('Saving to', name)
            experiment.save_trackers_to_json(name)
    

def is_of_interest(setting):
    
    """Determines whether a given setting is of interest.

    Args:
        setting (Dict): The current setting as a dictionary.

    Returns:
        bool: True if the setting is of interest, False otherwise.
    """
    if setting['dependent_last_tau']:
        if not setting['tau_is_trainable']:
            return False
        if setting['tau_l2_factor'] is not None:
            return False
        if setting['l1_reg_for_taus'] is not None:
            return False
        if 'adaptive_pruning' in setting.keys() and setting['adaptive_pruning']:
            return False
    if not setting['tau_is_trainable']:
        return setting['tau_l2_factor'] is None and setting['l1_reg_for_taus'] is None
    return True   

if __name__ == '__main__':
    main_res_net()
    main_frac_dnn()