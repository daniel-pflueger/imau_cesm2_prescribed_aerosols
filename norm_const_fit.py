#
# Daniel Pflueger, created 2021-12-03
# d.pfluger@uu.nl
#


from scipy.optimize import curve_fit
import expected_fields as xf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Fitting functions

def fit_one_norm_const(n_x,n_y,fit_options):
    '''
    Fit n_y to n_x (sub-routine of fit_norm_const)
    '''

    popt, pcov = curve_fit(fit_options['fit_func'], n_x, n_y, p0=fit_options['p0'],**fit_options['kwargs'])

    return {'popt': popt , 'pcov': pcov}

def fit_norm_const(results,main_name,options={}):
    '''
    Fit normalization constants stored in results against a chosen main field.

    Parameters:
    ---
    results (dict): Result dictionary as obtained by xf.load_exp
    main_name (string): Name of field that provides x values to the fit
    options (dict): Dictionary with same keywords as results. Items are dictionaries with fit options
        tbounds (list of int, optional): Time bounds over which to perform fit (e.g. [2050,2100] will only consider data between 2050 and 2100)
        fit_func (function): Function of form f(x,*args) used in fit
        p0 (list of float, optional): Initial guess of fit
        kwargs (dict): additional keyword arguments passed to scipy.optimize.curve_fit

    Returns:
    ---
    dict: Dictionary with keywords from results apart from main_field containing fit parameter arrays (popt) and covariances (pcov)
    '''

    # Extract norm consts from input results
    main_n = results[main_name]['n']
    names = list(results.keys())
    names.remove(main_name)
    results_n = {}
    for name in names:
        results_n[name] = results[name]['n']

    assert main_name in results.keys(), 'main_name must be field name in results'
    # Check that options input is properly defined
    assert isinstance(options,dict), 'options must be dictionary'
    if options == {}:
        # handle default case
        for name in names:
            options[name] = {'tbounds': None, 'p0': None, 'fit_func': None}

    # Apply tbounds/set kwargs to None if not specified/set p0 to None if not specified
    for name,option in options.items():
        if 'tbounds' in option.keys():
            assert option['tbounds'][0]<option['tbounds'][1], 'tbounds must be ordered'
        else:
            option['tbounds'] = [main_n.year[0].data,main_n.year[-1].data]
        ts = slice(str(option['tbounds'][0]),str(option['tbounds'][1]))
        # apply tbounds
        results_n[name] = results_n[name].sel(year=ts)
        if not ('p0' in option.keys()):
            option['p0'] = None
        if not ('fit_func' in option.keys()):
            option['fit_func'] = fit_func_power_law
        if not ('kwargs' in option.keys()):
            option['kwargs'] = {}

    result_dict = {}
    # iterate over all fields to fit to main norm const
    for name, norm in results_n.items():
        # if necessary, the main norm constant is restricted
        main_n_bdd = main_n.sel(year=slice(str(options[name]['tbounds'][0]),str(options[name]['tbounds'][1])))
        result_dict[name] = fit_one_norm_const(main_n_bdd,norm,options[name])
        result_dict[name]['ts'] = np.array(results_n[name].year)

    return result_dict

def fit_func_power_law(x,a,b,c):
    return a+b*x**c

def fit_func_power_law_shift(x,a,b,c,d):
    return a+b*(x-d)**c

def fit_func_saturate(x,a,b,c,d):
    return a+b*np.exp(c*(x-d))

# Obtaining norm constant from fit
def all_norms_from_main_field(n_main,path):
    '''
    Reconstruct of all fields from normalization of main field
    using fit results with custom fitting functions

    Implementation note: This function should be backwards-compatible with its previous implementation.
    Therefore, no adjustments to PIcontrol_exp_fields.py is necessary.

    Parameters:
    ---
    n_main (float): Normalization of main field
    path (str): Location of fit parameters, e.g. './fit_test_1/'

    Returns:
    ---
    dict: Reconstructed normalization constants
    '''

    assert path[-1]=='/', 'Path must end with /'

    fit_results, options = load_fit(path)
    n_results = {}
    for field in fit_results:
        fit_func = options[field]['fit_func']
        n_results[field] = fit_func(n_main,*fit_results[field]['popt'])

    return n_results

# I/O

def save_fit(fit_results,path,options=None):
    '''
    Save fit results obtained from fit_results to

    Parameters:
    ---
    fit_results (dict): Output of fit_norm_const to be saved
    path (str): Save directory, e.g. './fit_test_1/'
    options (str, default=None): Fit options supplied to fit_norm_const. If None (default), the fit options are not saved
    '''

    assert path[-1]=='/', 'Path must end with /'

    if not os.path.exists(path):
        print(f'Creating directory {path}')
        os.mkdir(path)

    with open(f'{path}params.pkl','wb') as save_file:
        pickle.dump(fit_results,save_file)
        if options:
            with open(f'{path}fit_options.pkl','wb') as options_file:
                # Note: when reloading the norm_const_fit module, pickling might fail
                # see https://stackoverflow.com/questions/1412787/picklingerror-cant-pickle-class-decimal-decimal-its-not-the-same-object
                pickle.dump(options,options_file)

def load_fit(path):
    '''
    Load fit results and options saved with save_fit

    Parameters:
    ---
    path (str): Save path used by save_fit. E.g. './fit_test_1/'

    Returns:
    ---
    dict (if fit options not found in path): Fit result dictionary
    (dict,dict) (if fit options found in path): Fit dictionary; Options dictionary
    '''

    assert os.path.exists(path), f'Path {path} does not exist.'
    assert os.path.exists(f'{path}params.pkl'), f'Fit parameter file {path}params.pkl does not exist.'

    with open(f'{path}params.pkl','rb') as save_file:
        fit_results = pickle.load(save_file)
        for field in fit_results:
            if 'ts' in fit_results.keys():
                fit_results[field].pop('ts')
        if os.path.exists(f'{path}fit_options.pkl'):
            with open(f'{path}fit_options.pkl','rb') as options_file:
                options = pickle.load(options_file)
            return (fit_results, options)
        else:
            return fit_results

# Visualization

def plot_fit_results(results,fit_results,options,main_field):
    '''
    Display the results of fit_norm_const as several rows of plots. This function is useful to test the quality of fits.

    Parameters:
    ---
    results (dict): Result dictionary obtained from xf.load_exp
    fit_results (dict): Output of fit_norm_const
    options (dict): Fit option dictionary supplied to fit_norm_const
    main_field (str): Name of main field, e.g. 'AODVISstdn'
    '''
    nfields = len(fit_results.keys())
    fig, axes = plt.subplots(nrows=nfields, figsize=(3,nfields*4))

    for ax,(name,result) in zip(axes,fit_results.items()):
        print(name)
        ax.set_ylabel(name)
        ax.set_xlabel(main_field)
        ax.plot(results['AODVISstdn']['n'],results[name]['n'])
        ax.plot(results['AODVISstdn']['n'],options[name]['fit_func'](results['AODVISstdn']['n'],*result['popt']),linestyle='dashed',color='C0')
