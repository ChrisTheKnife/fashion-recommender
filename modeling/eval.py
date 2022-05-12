from logging import getLogger
import pandas as pd
import numpy as np
import warnings
import mlflow
import sys, os
from datetime import datetime
from config import TRACKING_URI, EXPERIMENT_NAME

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

class DuplicateCustomerError(Exception):
    """ Exception will be raised when customers are appearing
    several times in the submission file."""
    pass

def prepare_data(sub_file, start_date, end_date):
    """Loads a given submission file and adds a ground_truth column, containing
       the list of item_id that the user actually bought during a given time period. 

    Args:
        sub_file (str, optional): Path and name of submission csv.
        start_date (str, optional): First day of evaluation period as datetime.
        end_date (str, optional): Last day of evaluation period as datetime.

    Returns:
        DataFrame: Dataframe of all customers that have bought something during the validation
        period. Index is the customer_id, prediction column gives the predicted item ids and
        ground_truth gives actual item ids (as whitespace-separated string).
    """

    logger.info("Preparing the data...")

    # load submission
    sub = pd.read_csv(sub_file, index_col=0)
    
    # check for duplicate predictions
    if len(sub.index) > len(sub.index.unique()):
        raise DuplicateCustomerError('Some customers are appearing multiple times in your submission file. Stopping...')

    # get rid of empty and nan predictions
    n_pred = len(sub)
    sub.replace('', np.NaN, inplace=True)
    sub.dropna(axis=0, inplace=True)
    n_after = len(sub)
    if n_pred > n_after:
        logger.warning(f'You have empty predictions and or nan predictions in your submission. These will not be scored. Number of dropped rows: {n_pred-n_after}')
    # load transactions
    df_trans = pd.read_csv('data/transactions_train.csv', parse_dates=[0], 
                           dtype={'article_id':'string'})

    # initially, drop all customers that made no purchase during the test period
    test_data = df_trans.query('t_dat >= @start_date and t_dat <= @end_date').copy()
    scored_customers = test_data.customer_id.unique()

    # reduce to include only customers that were actually predicted
    predicted_customers = sub.index.unique()
    predicted_and_scored = np.intersect1d(np.array(predicted_customers), np.array(scored_customers))
    # utter warning if not all scored customers were predicted
    if len(predicted_and_scored) < len(scored_customers):
        percentage_predicted = len(predicted_and_scored)/len(scored_customers) * 100.0
        logger.warning(f'You have not predicted all customers that bought something during the validation period. You have scored only {percentage_predicted} of all relevant customers.')
    sub = sub.loc[predicted_and_scored]

    # now collect item ids that were actually bought during test
    test_data.loc[:, 'article_id'] = test_data.article_id.apply(lambda i: str(i))
    ground_truth = test_data.groupby('customer_id').article_id.agg(lambda c: " ".join(c))
    # add that data as ground_truth column to the submission df
    sub['ground_truth'] = sub.index.map(mapper=ground_truth)
    logger.info("Done.")
    
    return sub

def average_precision_at_k(predicted, actual, cutoff=12):
    """Computes hitrate (precision) and average precision@k for given k and lists of predicted and actual
       item ids.

    Args:
        predicted list: list of predicted (recommended) item ids
        actual list: list of item ids bought during validation period
        cutoff (int, optional): Cutoff k to compute MAP@k. Defaults to 12.

    Returns:
        float: Average precision at cutoff k.
        float: Hitrate at k.
    """
    n_pred = len(predicted)
    if n_pred > cutoff:
        predicted = predicted[0:cutoff]
        n_pred = cutoff
    n_true = len(actual)
    avg_prec = 0.0
    how_many_of_k = 0
    for k in range(min(n_pred, cutoff)):
        # has the k-th item been correctly predicted?
        is_relevant = int((predicted[k] in actual) and (predicted[k] not in predicted[:k]))
        # track how many of the k items have been correctly predicted in total
        how_many_of_k += is_relevant
        # compute precision at cutoff k (=#relevant_recommendations/#of_recommendations)
        avg_prec += how_many_of_k/(k+1) * is_relevant
        # compute hitrate (precision)
    hitrate = how_many_of_k / n_true
    # normalize with number of actually bought items (or cutoff)
    return avg_prec / min(n_true, cutoff), hitrate

def mean_average_precision_at_k(sub, cutoff=12):
    """Computes mean hitrate (precision) and mean average precision at cutoff k given a submission, ground truth and
       cutoff.

    Args:
        sub (DataFrame): DataFrame as returned by prepare_data.
        cutoff (int, optional): Cutoff value k. Defaults to 12.

    Returns:
        float: mean average precision at cutoff k
        float: mean precision in percent
    """
    # compute mean average precision (at given cutoff)
    mean_avg_prec = 0.0
    mean_hitrate = 0.0
    # iterate over all test customers and sum up contributions 
    for values in sub.itertuples():
        predicted = values.prediction.split(' ')
        actual = values.ground_truth.split(' ')
        mean_avg_prec_inc, mean_hitrate_inc = average_precision_at_k(predicted, actual, cutoff)
        mean_avg_prec += mean_avg_prec_inc
        mean_hitrate += mean_hitrate_inc
    # normalize with number of customers
    return mean_avg_prec / len(sub), mean_hitrate / len(sub)

def run_evaluation(sub_file, start_date, end_date, cutoff, run_name):
    """Load a given submission and evaluate it using the mean average
       precision at cutoff k. Log metrics with mlflow.

    Args:
        sub_file (DataFrame): DataFrame containing predicted and actual item ids for test customers.
        start_date (datetime): Start of validation period.
        end_date (datetime): End of validation period.
        cutoff (int): Chosen cutoff k.
        run_name (string): Run name for mlflow.

    Returns:
        float: mean average precision at chosen cutoff
    """
    logger.info(f'Computing mean average precision for submission file {sub_file}.')
    logger.info(f'Selected test period: {start_date} -- {end_date}')
    logger.info(f'Selected cutoff k: {cutoff}')

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name): 
        # load data
        sub_data = prepare_data(sub_file, start_date=start_date, end_date=end_date)
        # compute MAP@cutoff
        logger.info('Computing hitrate and mean average precision...')
        map_at_k, mean_hitrate = mean_average_precision_at_k(sub_data, cutoff=cutoff)
        logger.info(f'Done. MAP@{cutoff}: {map_at_k} Mean Hitrate@{cutoff}: {mean_hitrate}')
    
        mlflow.log_metrics({f'MAP_at_{cutoff}':map_at_k, f'mean_hitrate_at_{cutoff}':mean_hitrate})
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'k': cutoff,
            'submission_name': 'test'}
    
        mlflow.log_params(params)
    return map_at_k


if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    # check command line arguments
    try:
        sub_file = sys.argv[1]
        if not os.path.exists(sub_file):
            logger.error('Input file does not seem to exist. Exiting...')
            raise(FileNotFoundError)
    except Exception:
        logger.warning('No input file provided. Using default.')
        sub_file = 'data/baseline_submission.csv'
    
    run_name = os.path.basename(sub_file).split('_')[0]

    try:
        start_date = datetime.strptime(sys.argv[2], '%Y-%m-%d')
        end_date = datetime.strptime(sys.argv[3], '%Y-%m-%d')
    except Exception:
        logger.warning('Either start_date and/or end_date are missing or malformed. Using default values instead.')
        start_date=datetime.strptime('2020-09-16', '%Y-%m-%d') 
        end_date=datetime.strptime('2020-09-22', '%Y-%m-%d') 
    
    try: 
        cutoff = int(sys.argv[4])
    except Exception:
        logger.warning('Cutoff value is missing or malformed. Using default value of 12.')
        cutoff=12

    run_evaluation(sub_file=sub_file, start_date=start_date, end_date=end_date, 
                   cutoff=cutoff, run_name=run_name)