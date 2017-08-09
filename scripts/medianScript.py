import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print('Let\'s explore this data.')

print('Reading data...')
key_1 = pd.read_csv('../input/key_1.csv')
train_1 = pd.read_csv('../input/train_1.csv')
ss_1 = pd.read_csv('../input/sample_submission_1.csv')

print ('Data has been read')

# The train file's first column is Page, and the rest of the columns have date names
# The dataset doesn't distinguish between no data and no visits.
# So we have to choose between 0 and median, essentially,
# before going on to make our predictions.

# The output shows that the key file has just two columns: Page, and Id
# This is what we use to make our predictions

# axis=1 means that we are computing the median along each row

# https://stackoverflow.com/questions/33058590/pandas-dataframe-replacing-nan-with-row-average

print ('filling nan with median by row in train_1 dataframe')
df = train_1.drop('Page', axis=1)
df.T.fillna(df.median(axis=1)).T
# array = df.values
array = df.values[:, -70:]

print ('nan values have been filled, taking median again')
visits = np.nan_to_num(np.round(np.nanmedian(array, axis=1)))
print ('Visits prediction has been made')


print ('Now we build our submission')
# for some reason we drop the last 11 letters of the pages
ids = key_1.Id.values
pages = key_1.Page.values
d_pages = {}
for id, page in zip(ids, pages):
    d_pages[id] = page[:-11]

pages = train_1.Page.values
# Now we put our predicted values into our new dictionary
d_visits = {}
for page, visits_number in zip(pages, visits):
    d_visits[page] = visits_number

print('Modifying sample submission...')
# Take the values of the Id and the Visits columns
ss_ids = ss_1.Id.values
ss_visits = ss_1.Visits.values

# enumerate is a Python method that loops over the index and item
for i, ss_id in enumerate(ss_ids):
    ss_visits[i] = d_visits[d_pages[ss_id]]

print('Saving submission...')
# Put in a DataFrame again to use the to_csv method
subm = pd.DataFrame({'Id': ss_ids, 'Visits': ss_visits})
subm.to_csv('submission.csv', index=False)

print('')
print('****************************')
print('And that\'s all she wrote')