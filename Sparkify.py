# import libraries
from pyspark.sql import SparkSession 
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc
from pyspark.sql.functions import asc
from pyspark.sql.functions import sum 

import datetime
import calendar

from pyspark.sql import Window
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import seaborn as sb
%matplotlib inline

import re
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import regexp_extract, col,  regexp_replace, split
from pyspark.sql.functions import create_map, lit
from itertools import chain


# ML imports
from pyspark.ml.feature import Normalizer, StandardScaler, VectorAssembler, MinMaxScaler, RobustScaler
from pyspark.ml.classification import LogisticRegression, GBTClassifier, LinearSVC, DecisionTreeClassifier, RandomForestClassifier, NaiveBayes
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def churn_down_id(df): 
     """
    Return sparkify data array of user ID that churned or downgraded.
    Parameters
    -----------
        df: DataFrame
    returns
    -------
        churn_ar: Array
        down_ar: Array
    """
    df1 = df.toPandas().dropna(subset=['firstName']) 
    churn_ar = df1['userId'][df1['page'] == 'Cancellation Confirmation'].unique()
    subD = df1['userId'][df1['page'] == 'Submit Downgrade'].unique()
    subUp = df1['userId'][df1['page'] == 'Submit Upgrade'].unique()
    UD = list(set(subD) & set(subUp)) 
    df_ud = df1[(df1['userId'].isin(UD)) & (df1['page'].isin(['Submit Upgrade', 'Submit Downgrade']))].drop_duplicates(subset='userId', keep='last')[['userId', 'page']]
    df_d = df_ud[df_ud['page'] == 'Submit Downgrade']['userId'].unique()  
    down = df1[(~df1['userId'].isin(UD)) & (df1['userId'].isin(subD))]['userId'].unique()
    down_ar = np.concatenate([df_d, down])
    return churn_ar, down_ar 

def data_prep(df, churn_ar, down_ar): 
    """
    Return sparkify data columns of churn and downgrade as well as rename of OS and states
    Parameters
    -----------
        df: DataFrame
        churn_ar: Array
        down_ar: Array
    returns
    -------
        df_prep: DataFrame
    """
    # downgrade, churn, end_level
    downgrade  = udf(lambda x: 1 if x in down_ar else 0, IntegerType())
    churn  = udf(lambda x: 1 if x in churn_ar else 0, IntegerType())
    end_level  = udf(lambda x: 'free' if x in down_ar else 'paid', StringType())
    df = df.withColumn('downgrade', downgrade('userId')) 
    df = df.withColumn('churn', churn('userId'))
    df = df.withColumn('end_level', end_level('userId'))

    # OS rename 
    df = ddf.withColumn('OS', split(col("exp"), ";").getItem(0))

    os_general = {'Macintosh': 'Mac', 'Windows NT 6.1': 'Windows', 'Windows NT 6.3': 'Windows', 
              'iPad': 'Mac',  'iPhone': 'Mac', 'compatible': 'Windows', 'X11': 'Linux', 
              'Windows NT 5.1': 'Windows', 'Windows NT 6.2': 'Windows', 'Windows NT 6.0': 'Windows'}
    os_specific = {'Macintosh': 'Mac', 'Windows NT 6.1': 'Windows 7', 'Windows NT 6.3': 'Windows 8.1', 
              'Ipad': 'Mac',  'Iphone': 'Mac', 'compatible': 'Windows 7', 'X11': 'Linux', 
              'Windows NT 5.1': 'Windows XP', 'Windows NT 6.2': 'Windows 8.0', 'Windows NT 6.0': 'Windows Vista'} 

    df = df.withColumn('gen_os', df['OS'])
    df = df.replace(to_replace=os_general, subset=['gen_os'])
    df = df.withColumn('spe_os', df['OS'])
    df = df.replace(to_replace=os_specific, subset=['spe_os']) 

    # Location rename 
    df = df.withColumn('state', split(col('location'), ',').getItem(1))

    return df

def dummy_var(df): 
    """
    Returning sparkify categoric variable from dummy processing into one dataframe
    Parameters
    -----------
        df: DataFrame
    returns
    -------
        df_cate: DataFrame
    """ 
    # gender into dummy_var 
    gen_dict = {'M': '1', 'F': '0'}
    df = df.replace(to_replace=gen_dict, subset=['gender'])
    df = df.withColumn('gender', df['gender'].cast(IntegerType()))
    gender_feat = df.select(['userId', 'gender']).dropDuplicates() 

    # OS into dummy_var 
    os_name = df.select('spe_os').distinct().rdd.flatMap(lambda x: x).collect()
    os_exp = [F.when(F.col('spe_os') == osdict, 1).otherwise(0).alias(osdict) for osdict in os_name]
    os_feat = df.select('userId', *os_exp).dropDuplicates() 

    # states into dummy_var 
    state_name = df.select('state').distinct().rdd.flatMap(lambda x: x).collect()
    state_exp = [F.when(F.col('state') == statedict, 1).otherwise(0).alias(statedict) for statedict in state_name]
    state_feat = df.select('userId', *state_exp).dropDuplicates() 

    # end_level into dummy_var 
    end_dict = {'paid': '0', 'free': '1'}
    df = df.replace(to_replace=end_dict, subset=['end_level'])
    df = df.withColumn('end_level', df['end_level'].cast(IntegerType()))
    end_feat = df.select(['userId', 'end_level']).dropDuplicates()

    df_cate = gender_feat.join(os_feat,'userId','outer') \
        .join(state_feat,'userId','outer') \
        .join(end_feat,'userId','outer')  

    df_cate = df_cate.withColumnRenamed("Windows 8.1", "Windows 81")
    df_cate = df_cate.withColumnRenamed("Windows 8.0", "Windows 80")

    return df_cate

def numeric_var(df): 
    """
    Returning sparkify numeric variable from standardizing into one dataframe
    Parameters
    -----------
        df: DataFrame
    returns
    -------
        scaled_data: DataFrame
    """ 
    # Song length in all of their sessions 
    length_df = df.filter(df.page=='NextSong').select('userId', 'sessionId', 'length')
    length_df = length_df.withColumn('sessions', (length_df.length/3600))
    length_df = length_df.groupBy('userId', 'sessionId').sum('sessions')
    length_df = length_df.groupBy('userId').agg(F.sum('sum(sessions)').alias('ses_hours')).na.fill(0) 

    # page numeric features 
    page_df = df.groupBy('userId').pivot('page').count().na.fill(0)
    page_df = page_df.select(['userId', 'Add Friend', 'Add to Playlist', 'Thumbs Down', 'Thumbs Up'])

    # user subs age 
    start_sub = df.select('userId', 'registration').dropDuplicates().withColumnRenamed('registration', 'start')
    end_sub = df.groupBy('userId').max('ts').withColumnRenamed('max(ts)', 'end')
    sub_df = start_sub.join(end_sub,'userId')
    ticks_per_day = 1000 * 60 * 60 * 24 
    sub_df = sub_df.select('userId', ((sub_df.end-sub_df.start)/ticks_per_day).alias('Subs Duration in Days'))

    # Standardize the numeric features 
    df_num = length_df.join(page_df,'userId','outer') \
        .join(sub_df,'userId','outer') \

    columns = df_num.columns[1:]
    assemblers = [VectorAssembler(inputCols=[col],outputCol=col+'_vect') for col in columns]
    scalers = [RobustScaler(inputCol=col+'_vect', outputCol=col+'_scaled', withScaling=True, withCentering=False,
                        lower=0.25, upper=0.75) for col in columns]
    pipeline = Pipeline(stages=assemblers + scalers)
    scaledmodel = pipeline.fit(df_num)
    scaled_data = scaledmodel.transform(df_num)

    scaled_data = scaled_data.select(['userId', 'ses_hours_scaled',
     'Add Friend_scaled',
     'Add to Playlist_scaled',
     'Thumbs Down_scaled',
     'Thumbs Up_scaled',
     'Subs Duration in Days_scaled'])

    return scaled_data

def join_data(df, df_cate, scaled_data): 
    """
    Returning sparkify ready dataframe by joining categoric, numeric and results columns
    Parameters
    -----------
        df: DataFrame
        df_cate: DataFrame 
        scaled_data: Dataframe
    returns
    -------
        df_down1: DataFrame
        df_churn1: DataFrame
    """ 
    churn_df = df.select(['userId', col('churn').alias('label')]).dropDuplicates()
    down_df = df.select(['userId', col('downgrade').alias('label')]).dropDuplicates()
    # join churn and vectorize it 
    df_churn = scaled_data.join(df_cate,'userId','outer') \
        .join(churn_df,'userId','outer') \
        .drop('userId')

    cols = df_churn.columns
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    df_churn1 = assembler.transform(df_churn) 

    # join downgrade and vectorize it 
    df_down = scaled_data.join(df_cate,'userId','outer') \
        .join(down_df,'userId','outer') \
        .drop('userId')

    cols = df_down.columns
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    df_down1 = assembler.transform(df_down) 

    return df_churn1, df_down1 

def data_split(df_churn, df_down): 
    """
    Returning sparkify datasets that are splitted into train, valid and test
    Parameters
    -----------
        df_churn: DataFrame
        df_down: DataFrame 
    returns
    -------
        trainc: DataFrame
        validc: DataFrame
        testc: DataFrame
        traind: DataFrame
        validd: DataFrame
        testd: DataFrame
    """
    trainc, validc, testc = df_churn.randomSplit([0.6, 0.2, 0.2], seed=139)
    traind, validd, testd = df_down.randomSplit([0.6, 0.2, 0.2], seed=139)

    return trainc, validc, testc, traind, validd, testd

def model_valid(algo, df_train, df_valid, title): 
    """
    Returning sparkify dataframe that are trained by train set and tested against validation set
    Parameters
    -----------
        algo: Strings 
        df_train: DataFrame
        df_test: DataFrame 
        title: Strings
    returns
    -------
        cvModel: Cross validator model
    """
    algorithm = algo()

    paramGrid = ParamGridBuilder().addGrid(gbt.maxBins,[16, 32]).addGrid(gbt.maxDepth,[4, 5]).build()

    crossval = CrossValidator(estimator=algorithm,
                              evaluator=BinaryClassificationEvaluator(metricName='areaUnderPR'), 
                              estimatorParamMaps=paramGrid,
                              numFolds=3)
    cvModel = crossval.fit(df_train)
    results = cvModel.transform(df_valid)
    evaluatorb = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
    evaluatorm = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    print(title)
    print('area under ROC: %f' % evaluatorb.evaluate(results, {evaluatorb.metricName: "areaUnderROC"}))
    print('area under PR: %f' % evaluatorb.evaluate(results, {evaluatorb.metricName: "areaUnderPR"}))
    print('Accuracy: %f' % evaluatorm.evaluate(results, {evaluatorm.metricName: "accuracy"}))
    print('F-1 Score: %f' % evaluatorm.evaluate(results, {evaluatorm.metricName: "f1"}))
    print('wPrecision: %f' % evaluatorm.evaluate(results, {evaluatorm.metricName: "weightedPrecision"}))
    print('wRecall: %f' % evaluatorm.evaluate(results, {evaluatorm.metricName: "weightedRecall"}))
    print(cvModel.avgMetrics, paramGrid)

    return cvModel

def model_test(cvModel, df_test)
    """
    Returning sparkify trained datasets and test it against test datasets with best parameters
    Parameters
    -----------
        cvModel: Cross validator model
        df_test: DataFrame 
    returns
    -------
        results: dataframe
    """
    best_model = cv_model.bestModel
    results = best_model.transform(df_test)
    evaluatorb = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
    evaluatorm = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    print(title)
    print('area under ROC: %f' % evaluatorb.evaluate(results, {evaluatorb.metricName: "areaUnderROC"}))
    print('area under PR: %f' % evaluatorb.evaluate(results, {evaluatorb.metricName: "areaUnderPR"}))
    print('Accuracy: %f' % evaluatorm.evaluate(results, {evaluatorm.metricName: "accuracy"}))
    print('F-1 Score: %f' % evaluatorm.evaluate(results, {evaluatorm.metricName: "f1"}))
    print('wPrecision: %f' % evaluatorm.evaluate(results, {evaluatorm.metricName: "weightedPrecision"}))
    print('wRecall: %f' % evaluatorm.evaluate(results, {evaluatorm.metricName: "weightedRecall"}))

    return results


def ExtractFeature(featureImp, dataset, featuresCol):
    """
    Returning a dataframe that consists of features weight according to the trained sets and results from test set
    Parameters
    -----------
        featureImp: Array
        dataset: DataFrame
        featuresCol: Array 
    returns
    -------
        varlist: dataframe
    """
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

def feat_import(cvModel, results): 
    """
    Returning a dataframe that consists of features that affected the model
    Parameters
    -----------
        cvModel: Cross validator model
        results: DataFrame
    returns
    -------
        df_red: dataframe
    """
    df_red = ExtractFeature(cvModel.featureImportances, results, "features")
    df_red = df_red[df_red['score'] >= 0.01].reset_index().drop(['index', 'idx'], axis=1)
    df_red['cumsum'] = df_red['score'].cumsum() 
    df_red['name'] = df_red['name'].str.rsplit('_', n=1).str[0] 
    print(df_red)

    return df_red

def reduce_model(df_red, df_churn1): 
    """
    Returning a dataframe which selected features affected 75% cumulative weight of the model
    Parameters
    -----------
        df_red: Dataframe
        df_churn1: DataFrame
    returns
    -------
        df_red1: dataframe
    """
    finalcol = np.array(df_red[df_red['cumsum'] <= 0.75]['name'])
    feat_col = np.append(finalcol,['label'])
    feat_col1 = finalcol

    df_red = df_churn1.select(featc_col) 
    assembler = VectorAssembler(inputCols=featc_col1, outputCol="features") 
    df_red1 = assembler.transform(df_red)

    return df_red1


if __name__=='__main__':
    # create a Spark session
    spark = SparkSession.builder.appName(
        "Sparkify").getOrCreate()
    data_path = "mini_sparkify_event_data.json"
    df = spark.read.json(data_path)
    churn_ar, down_ar = churn_down_id(df)
    df = data_prep(df, churn_ar, down_ar)
    df_cate = dummy_var(df)
    scaled_data = numeric_var(df)
    df_churn1, df_down1 = join_data(df, df_cate, scaled_data)
    trainc, validc, testc, traind, validd, testd = data_split(df_churn1, df_down1)
    cvModelc = model_valid(GBTClassifier, trainc, validc, 'Gradient Boosted Trees Metrics Churn:')
    cvModeld = model_valid(GBTClassifier, traind, validd, 'Gradient Boosted Trees Metrics Downgrade:')
    results_bestc = model_test(cvModelc, testc)
    results_bestd = model_test(cvModeld, testd)
    feat_c = feat_import(cvModelc, results_bestc)
    feat_d = feat_import(cvModeld, results_bestd)
    df_redc = reduce_model(feat_c, df_churn1)
    df_redd = reduce_model(feat_d, df_down1)
    trainrc, validrc, testrc, trainrd, validrd, testrd = data_split(df_redc, df_redd)
    cvModelredc = model_valid(GBTClassifier, trainrc, validrc, 'Gradient Boosted Trees Metrics Red Churn:')
    cvModelredd = model_valid(GBTClassifier, trainrd, validrd, 'Gradient Boosted Trees Metrics Red Downgrade:')
    results_bestc = model_test(cvModelredc, testrc)
    results_bestd = model_test(cvModelredd, testrd)
