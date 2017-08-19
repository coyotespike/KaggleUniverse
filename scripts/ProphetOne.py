import numpy as np
import pandas as pd
from fbprophet import Prophet
import math

print('Reading data...')
key_1 = pd.read_csv('../input/key_1.csv')
train_1 = pd.read_csv('../input/train_1.csv')
ss_1 = pd.read_csv('../input/sample_submission_1.csv')

print('Preprocessing...')

print('Processing...')
ids = key_1.Id.values
pages = key_1.Page.values


print('key_1...')
d_pages = {}
for id, page in zip(ids, pages):
    d_pages[id] = page[:-11]

print('train_1...')
pages = train_1.Page.values
# drop signifies we are deleting the row or column, axis=1 means a column
# doesn't mutate in place by default, returning new df
# then we appear to do nothing more than take the average of the page visits on each date
# finally we impute median values to NaN entries, and drop the last 56 entries
# so what's in those?

# score: 45.7
visits = np.nan_to_num(np.round(np.mean(train_1.drop('Page', axis=1).values, axis=1))) # Version 2 score: 64.8

# Now we build our submission
d_visits = {}
for page, visits_number in zip(pages, visits):
    d_visits[page] = visits_number

# what does the visits prediction look like at this point?
# it is just page names and visits, apparently
# if this script just uses the mean/median for all dates, then it can just
## use the page name as a key.
## Thus need to do more processing a la peter to match up dates.


print('Modifying sample submission...')
ss_ids = ss_1.Id.values
ss_visits = ss_1.Visits.values

for i, ss_id in enumerate(ss_ids):
    ss_visits[i] = d_visits[d_pages[ss_id]]

print('Saving submission...')
subm = pd.DataFrame({'Id': ss_ids, 'Visits': ss_visits})

# 45.7
df = train_1
droppedDF = df.drop(df.columns[0], axis=1)
droppedDF.columns = pd.to_datetime(droppedDF.columns, infer_datetime_format=True)
print(droppedDF.dtypes)
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
        myArr = key_1.Page.values
        foundIndex = np.nonzero(myArr == pages[i] + "_2017-01-01")[0][0]
        counter = 0
        for index, row in seriesPrediction.iterrows():
            ind = foundIndex + counter
            print ("we got to " + str(i))
            final = math.ceil(row['yhat'])
            print ("we predicted " + str(final))
            print ("our index was " + str(ind))
            subm.ix[ind, "Visits"] = final
            counter +=1

subm.to_csv('submission.csv', index=False)

# this script took about 8 minutes to hit 100
# estimate 40 minutes to do 500, then
# 9.15 to hit 116
