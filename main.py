import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import folium

# Load the original dataset
data = pd.read_csv("dataset.csv")

#Global dataset copies
geo_data = data.copy()   #Dataset copy for the geoplot and folium map
geo_data = geo_data[['lat', 'long', 'is_fraud']]    #Take only what we need for geographical display
geo_data = geo_data[geo_data['is_fraud'] != 0].copy().reset_index(drop=True)   #Reset index for entries

data_copy = data.copy() #Dataset copy for the calculate_most_likely_fraud_day function

frauded_data = data.copy()    #Dataset copy for the fraud count vs amount range plot
frauded_data = frauded_data[['amt', 'is_fraud']].astype(float)
frauded_data = frauded_data[frauded_data['is_fraud'] != 0].copy()
frauded_data = frauded_data[frauded_data['is_fraud'] != 0].copy().reset_index(drop=True)

fraud_data = data[['trans_date_trans_time', 'is_fraud']].copy() #Copy for the fraud rate over time plot

#Fraud count vs amount range
def count_vs_amount():
    amount_list = list(range(0, 1225, 10))    #amount range
    amount_fraud_list = [0] * len(amount_list)  #amount range counters

    #Iterate through the dataset then compare each entry with each amount in amount_list
    #When a row exceeds an amount,max reachable amount for that row found
    for i, row in frauded_data.iterrows():
        max_amount_reached = False #Flag for checking if the row reached the maximum amount in the list
        for j, amount in enumerate(amount_list):
            if row['amt'] > amount:
                continue
            else:
                max_amount_reached = True  #if max found increment amount position in counter list and exit row for loop
                if max_amount_reached:
                    amount_fraud_list[j] += 1
                    break

    # Plotting the fraud count against the amount ranges
    plt.plot(amount_list, amount_fraud_list)
    plt.xlabel('Amount ranges')
    plt.ylabel('Fraud count')
    plt.title('Fraud count vs Amount ranges')
    plt.show()

#Function for comparing the fraud rate between males and females
import numpy as np

def gender_comp(data):
    total_data_M = data[data['gender'] == 'M']
    total_data_F = data[data['gender'] == 'F']

    # Count the total number of transactions for each gender
    total_counts_M = total_data_M.shape[0]
    total_counts_F = total_data_F.shape[0]

    # Filter the dataset for fraudulent transactions for males and females
    filtered_data_M = data[(data['gender'] == 'M') & (data['is_fraud'] == 1)]
    filtered_data_F = data[(data['gender'] == 'F') & (data['is_fraud'] == 1)]

    # Count the number of fraudulent transactions for each gender
    fraud_counts_M = filtered_data_M.shape[0]
    fraud_counts_F = filtered_data_F.shape[0]

    # Create a bar plot to compare the total and fraudulent transactions for males and females
    categories = ['Male', 'Female']
    total_values = [total_counts_M, total_counts_F]
    fraud_values = [fraud_counts_M, fraud_counts_F]

    bar_width = 0.35
    index = np.arange(len(categories))

    plt.bar(index, total_values, bar_width, label='Total')
    plt.bar(index + bar_width, fraud_values, bar_width, label='Fraudulent')

    # Add the total number as annotations next to the bars
    for i, val in enumerate(total_values):
        plt.annotate(str(val), xy=(i, val), xytext=(0, 3), textcoords="offset points", ha='center')
    for i, val in enumerate(fraud_values):
        plt.annotate(str(val), xy=(i + bar_width, val), xytext=(0, 3), textcoords="offset points", ha='center')

    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Total vs Fraudulent Transactions by Gender')
    plt.xticks(index + bar_width / 2, categories)
    plt.legend()
    plt.show()


#Function for displaying the fraudulent data on the geoplot
def geoplot():
    # Load the shapefile for the United States
    us_shapefile_path = 'ne_110m_admin_1_states_provinces.shp'
    us_map = gpd.read_file(us_shapefile_path)
    assert 'lat' in geo_data.columns and 'long' in geo_data.columns

    # Create a GeoDataFrame for fraudulent entries
    geometry = [Point(xy) for xy in zip(geo_data['long'], geo_data['lat'])]
    fraud_gdf = gpd.GeoDataFrame(geo_data, geometry=geometry)
    # Plot the map
    fig, ax = plt.subplots(figsize=(12, 7))

    #Asign labels to the axis
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Plot the base map
    us_map.plot(ax=ax, color='white', edgecolor='black')
    # Plot fraudulent entries
    fraud_gdf.plot(ax=ax, color='red', markersize=20, alpha=0.5)
    plt.show()
    return fraud_gdf



#Function that caluclates the fraud rate over time
def fraud_rate():
    # Convert trans_date_trans_time to datetime variable type
    fraud_data['trans_date_trans_time'] = pd.to_datetime(fraud_data['trans_date_trans_time'], format='%m/%d/%Y %H:%M')
    # Sort the DataFrame by trans_date_trans_time in ascending order
    sorted_fraud_data = fraud_data.sort_values('trans_date_trans_time')
    #Calculate the sum of frauds for each individual day
    fraud_rate = fraud_data.groupby(fraud_data['trans_date_trans_time'].dt.date)['is_fraud'].sum()


    #Plot results
    plt.plot(fraud_rate.index, fraud_rate.values)
    plt.xlabel('Date')
    plt.ylabel('Fraud Rate')
    plt.title('Fraud Rate Over Time')
    plt.xticks(rotation=45)
    plt.show()

    high_fraud_days = fraud_rate[fraud_rate > 20]
    print(high_fraud_days)

    return sorted_fraud_data  #optional return (only used for the verifying)

def calculate_most_likely_fraud_day():
    dataset = data.copy()
    #Convert trans_date_trans_time to datetime variable type
    dataset['trans_date_trans_time'] = pd.to_datetime(dataset['trans_date_trans_time'], format='%m/%d/%Y %H:%M')
    #Create new column with the coresponding day of the week name for the date
    dataset['day_of_week'] = dataset['trans_date_trans_time'].dt.day_name()

    # Calculate the sum of frauds for each day of the week
    fraud_days = dataset[dataset['is_fraud'] == 1]['day_of_week'].value_counts()

    # Find the day with the maximum count
    most_likely_day = fraud_days.idxmax()
    return most_likely_day

#Type cast string features into integers
def feature_type_cast(data):
    #Type cast the features(because strings dont work with the model)
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col])

#Function for displaying the accuracy scores of the model
def print_accuracy_scores(accuracy, precision, recall, f1):
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

#Function for verifying the fraud_rate() functionality (for algorithm testing purposes)
def time_validation(year, month, day):
    fraud_rate()['year'] = fraud_rate()['trans_date_trans_time'].dt.year
    fraud_rate()['month'] = fraud_rate()['trans_date_trans_time'].dt.month
    fraud_rate()['day'] = fraud_rate()['trans_date_trans_time'].dt.day
    filtered_data = fraud_rate()[(fraud_rate()['year'] == year) & (fraud_rate()['month'] == month) & (fraud_rate()['day'] == day) & (fraud_rate()['is_fraud'] == 1)]
    print("Number of entries from the selected date:", len(filtered_data))

#Function for generating the floium map
def folium_map():
    fraud_json = geoplot().to_json()  #Converting the geoplot into a json file
    m = folium.Map(location=[geo_data['lat'].mean(), geo_data['long'].mean()], zoom_start=4.5)
    folium.GeoJson(fraud_json).add_to(m)
    m.save('fraud_map.html')

#Statistical functions call
count_vs_amount()
geoplot()
fraud_rate()
gender_comp(data)
print("Most likely day for frauds: ", calculate_most_likely_fraud_day())

#Optimize original dataset
unnecessary_columns = ["cc_num", "first", "last", "gender", "job", "dob", "zip", "lat", "long", "merch_lat",
                       "merch_long", "city_pop", "trans_num", "unix_time"]
data = data.drop(unnecessary_columns,axis = 1)
feature_type_cast(data)

#Split dataset into features and target
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Split the data into training and testing sets (20%test 80% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=True)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#Display accuracy scores
print_accuracy_scores(accuracy, precision, recall, f1)


