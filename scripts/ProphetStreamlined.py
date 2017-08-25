import numpy as np
import pandas as pd
from fbprophet import Prophet
import math

key_1 = pd.read_csv('../input/key_1.csv')
train_1 = pd.read_csv('../input/train_1.csv')

# 45.7

# Copy key_1 and add a column for visits
# Index key_1 by page name

# fill in with means
# overwrite as far as time allows with Prophet
# For each page in train_1, do our prediction overwriting for 60 columns

# After visits filled in, drop the page name column

keyPageArray = key_1.Page.values
df = train_1
droppedDF = df.drop(df.columns[0], axis=1)
droppedDF.columns = pd.to_datetime(droppedDF.columns, infer_datetime_format=True)
for i in range(250):
    series = droppedDF.iloc[i]
    # Prophet errors if the entire row is nan or 0
    if not series.isnull().all():
        miniDF = series.to_frame().reset_index()
    
        miniDF = miniDF.rename(columns={'index': 'ds', i: 'y'})
    
        # do we need to call Prophet() every time?
        m = Prophet()
        m.fit(miniDF)
        future = m.make_future_dataframe(periods=60)
        forecast = m.predict(future)
        
        # our forecast is completed. Take the desired values.
        seriesPrediction = forecast.tail(60)
        foundIndex = np.nonzero(keyPageArray == pages[i] + "_2017-01-01")[0][0]
        counter = 0
        for index, row in seriesPrediction.iterrows():
            final = math.ceil(row['yhat'])
            subm.ix[foundIndex + counter, "Visits"] = final
            counter +=1

subm.to_csv('submission.csv', index=False)
