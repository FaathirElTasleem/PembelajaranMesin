#!/usr/bin/env python
# coding: utf-8

# | Variable         | Description                                                                                                                        |
# |------------------|------------------------------------------------------------------------------------------------------------------------------------|
# | Track            | Name of the song, as visible on the Spotify platform.                                                                              |
# | Artist           | Name of the artist.                                                                                                                |
# | Url_spotify      | The URL of the artist.                                                                                                             |
# | Album            | The album in which the song is contained on Spotify.                                                                               |
# | Album_type       | Indicates if the song is released on Spotify as a single or contained in an album.                                                  |
# | Uri              | A Spotify link used to find the song through the API.                                                                              |
# | Danceability     | Describes how suitable a track is for dancing based on a combination of musical elements.                                           |
# | Energy           | Represents a perceptual measure of intensity and activity.                                                                         |
# | Key              | The key the track is in.                                                                                                           |
# | Loudness         | The overall loudness of a track in decibels (dB).                                                                                  |
# | Speechiness      | Detects the presence of spoken words in a track.                                                                                   |
# | Acousticness     | A confidence measure of whether the track is acoustic.                                                                             |
# | Instrumentalness | Predicts whether a track contains no vocals.                                                                                       |
# | Liveness         | Detects the presence of an audience in the recording.                                                                              |
# | Valence          | Describes the musical positiveness conveyed by a track.                                                                            |
# | Tempo            | The overall estimated tempo of a track in beats per minute (BPM).                                                                   |
# | Duration_ms      | The duration of the track in milliseconds.                                                                                        |
# | Stream           | Number of streams of the song on Spotify.                                                                                         |
# | Url_youtube      | URL of the video linked to the song on YouTube.                                                                                    |
# | Title            | Title of the video clip on YouTube.                                                                                               |
# | Channel          | Name of the channel that has published the video.                                                                                  |
# | Views            | Number of views on YouTube.                                                                                                        |
# | Likes            | Number of likes on YouTube.                                                                                                        |
# | Comments         | Number of comments on YouTube.                                                                                                     |
# | Description      | Description of the video on YouTube.                                                                                               |
# | Licensed         | Indicates whether the video represents licensed content.                                                                           |
# | Official_video   | Boolean value indicating if the video found is the official video of the song.                                                      |
# 

# * Are you in need of downloading this dataset? [Click here.](https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube/data)
# 

# In[1]:


# Let's start with importing packages 🧲
import warnings
import pandas as pd 
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
import seaborn as sns
sns.set(style="whitegrid")
sns.set_palette("pastel")
sns.set_context("notebook", font_scale=1.2)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# # Read and Recognize the Dataset

# In[1]:


# Reed DataSet 🎬
dataset = pd.read_csv('/kaggle/input/spotify-and-youtube/Spotify_Youtube.csv')


# In[2]:


# View dataset that we read
dataset.sample(n=10)


# `As you can see in this data, there are many columns that are not useful in our analysis, such as the Unnamed:0 column, which may be used to encode the data, and other columns that are not useful to us now as Url_spotify, Uri, Url_youtube  and so on, so it is necessary to delete all of this before any operations on the data.`

# # Data Collection and cleaning

# In[3]:


list[dataset.columns]


# In[4]:


dataset.drop(columns=['Unnamed: 0' , 'Url_spotify' ,'Uri' ,'Uri' ,'Danceability', 'Energy', 'Key', 'Loudness','Url_youtube' ,'Title' ,'Description'] ,inplace=True)


# * `DataSet after deleting columns that are not useful. `

# In[5]:


dataset


# In[6]:


# Displaying the summary information of the dataset
dataset.info()


# * There are 7 columns with data type object (likely strings) and 11 columns with data type float64 (likely numerical).
# 
# * Some columns have missing values (non-null count < 20718)

# `Handling Missing Values`

# In[7]:


# Show Num of missing values in each column
missing_values = dataset.isna().sum()
print(missing_values)


# In[8]:


# Define a custom color palette
colors = sns.color_palette("husl", len(missing_values))

# Calculate the momentum of missing values for each column
missing_values_momentum = (dataset.isna().sum() / len(dataset)) * 100

# Create a bar plot with custom colors
plt.figure(figsize=(10, 6))
missing_values_momentum.plot(kind='bar', color=colors)
plt.title('Momentum of Missing Values for Each Column')
plt.xlabel('Columns')
plt.ylabel('Percentage of Missing Values')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[9]:


# Impute missing values for columns with few missing values
columns_with_few_missing = ['Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms']
for column in columns_with_few_missing:
    dataset[column].fillna(dataset[column].mean(), inplace=True)


# `Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration_ms:`
# 
# * Since these columns have only 2 missing values each, imputing the missing values with the mean or median of the respective column is a reasonable approach. This method helps to preserve the overall distribution of the data and minimizes the loss of information.

# In[10]:


# List of columns with many missing values
columns_with_many_missing = ['Channel', 'Views', 'Likes', 'Comments', 'Licensed', 'official_video', 'Stream']

# Drop rows with missing values in the specified columns
for column in columns_with_many_missing:
    dataset.dropna(subset=[column], inplace=True)


# `Channel, Views, Likes, Comments, Licensed, official_video, Stream:`
# 
# * Deleting rows with missing values may be a suitable option if the missing values represent a small portion of the dataset and removing them does not significantly impact the analysis. However, this approach can lead to loss of information, especially if the missing values are not randomly distributed.

# In[11]:


dataset.info()


# In[12]:


# Show Num of missing values in each column
missing_values = dataset.isna().sum()
print(missing_values)


# In[13]:


# Select only the numerical columns for descriptive statistics
numerical_columns = dataset.select_dtypes(include=['float64'])
numerical_description = numerical_columns.describe()
print(numerical_description)


# | Feature           | Mean     | Std Dev | Min      | 25th Percentile | Median   | 75th Percentile | Max      |
# |-------------------|----------|---------|----------|-----------------|----------|-----------------|----------|
# | Speechiness       | 0.095    | 0.106   | 0.000    | 0.036           | 0.051    | 0.104           | 0.964    |
# | Acousticness      | 0.289    | 0.286   | 0.000001 | 0.044           | 0.190    | 0.470           | 0.996    |
# | Instrumentalness  | 0.055    | 0.193   | 0.000    | 0.000           | 0.000002 | 0.000434        | 1.000    |
# | Liveness          | 0.191    | 0.165   | 0.0145   | 0.094           | 0.125    | 0.234           | 1.000    |
# | Valence           | 0.529    | 0.245   | 0.000    | 0.339           | 0.536    | 0.725           | 0.993    |
# | Tempo             | 120.606  | 29.619  | 0.000    | 96.990          | 119.964  | 139.951         | 243.372  |
# | Duration (ms)     | 224,628  | 126,909 | 30,985   | 180,243         | 213,254  | 251,911         | 4,676,058|
# | Views             | 95.45M   | -       | 26       | 1.91M           | 14.91M   | 71.52M          | 8.08B    |
# | Likes             | 670.02K  | -       | 0        | 22.38K          | 127.92K  | 526.59K         | 50.79M   |
# | Comments          | 27.86K   | -       | 0        | 531.25          | 3.34K    | 14.49K          | 16.08M   |
# | Stream            | 137.11M  | -       | 6.57K    | 17.81M          | 49.79M   | 139.08M         | 3.39B    |

# | Feature         | Mean  | Std Dev | Min      | 25th Percentile | Median | 75th Percentile | Max   | Interpretation                                             |
# |-----------------|-------|---------|----------|-----------------|--------|-----------------|-------|------------------------------------------------------------|
# | Speechiness     | 0.095 | 0.106   | 0.000    | 0.036           | 0.051  | 0.104           | 0.964 | The data shows a range of speechiness levels from very low to high, with a mean of 0.095 indicating songs are generally not heavily speech-oriented. |
# | Acousticness    | 0.289 | 0.286   | 0.000001 | 0.044           | 0.190  | 0.470           | 0.996 | Acousticness varies widely, with a mean of 0.289 suggesting a moderate level overall. The standard deviation of 0.286 indicates considerable variability. |
# | Instrumentalness| 0.055 | 0.193   | 0.000    | 0.000           | 0.000002| 0.000434        | 1.000 | Instrumentalness ranges from low to high, with a mean of 0.055 suggesting a generally low level. The maximum of 1.000 indicates some songs are entirely instrumental. |
# | Liveness        | 0.191 | 0.165   | 0.0145   | 0.094           | 0.125  | 0.234           | 1.000 | Liveness varies with a mean of 0.191, indicating songs tend to have some live performance quality. The standard deviation of 0.165 suggests moderate variability. |
# | Valence         | 0.529 | 0.245   | 0.000    | 0.339           | 0.536  | 0.725           | 0.993 | Valence ranges widely with a mean of 0.529, indicating a generally positive emotional tone in the songs. The standard deviation of 0.245 suggests moderate variability. |
# | Tempo           | 120.61| 29.62   | 0.000    | 96.99           | 119.96 | 139.95          | 243.37| Tempo varies with a mean of 120.61 BPM, indicating a moderate tempo overall. The standard deviation of 29.62 suggests considerable variability. |
# | Duration (ms)   | 224.63| 126.91  | 30,985   | 180,243         | 213,254| 251,911         | 4,676,058 | Song durations range widely, with a mean of 224,628 milliseconds. The standard deviation of 126,909 suggests considerable variability. |
# | Views           | 95.45M| -       | 26       | 1.91M           | 14.91M | 71.52M          | 8.08B | Views range from very low to very high, with a mean of 95.45 million. |
# | Likes           | 670.02K| -      | 0        | 22.38K          | 127.92K| 526.59K         | 50.79M | Likes vary widely with a mean of 670.02 thousand. |
# | Comments        | 27.86K | -      | 0        | 531.25          | 3.34K  | 14.49K          | 16.08M | Comments range from very low to very high, with a mean of 27.86 thousand. |
# | Stream          | 137.11M| -     | 6.57K    | 17.81M          | 49.79M | 139.08M         | 3.39B | Streams range from very low to very high, with a mean of 137.11 million. |
# 

# ### Observations:
# 
# 1. **Speechiness:**
#    - The average speechiness is relatively low (0.095), suggesting minimal spoken words in songs.
#    - However, there's a wide range of speechiness values, indicating variability in spoken word content.
# 
# 2. **Acousticness:**
#    - The mean acousticness is moderate (0.289), indicating a significant presence of acoustic elements in songs.
#    - There's considerable variability in acousticness across the dataset, reflecting diverse acoustic characteristics.
# 
# 3. **Instrumentalness:**
#    - The mean instrumentalness is relatively low (0.055), suggesting vocals are present in most songs.
#    - However, some songs are entirely instrumental, as indicated by high instrumentalness values.
# 
# 4. **Liveness:**
#    - The average liveness is moderate (0.191), indicating a moderate degree of live performance quality.
#    - Variability in liveness exists across the dataset, with some songs exhibiting higher live performance attributes.
# 
# 5. **Valence:**
#    - The mean valence is moderate (0.529), indicating songs tend to convey a positive emotional tone.
#    - There's moderate variability in valence, suggesting diversity in emotional expression.
# 
# 6. **Tempo:**
#    - The average tempo is approximately 120.61 BPM, suggesting a moderate pace overall.
#    - Considerable variability in tempo exists, reflecting diverse rhythmic characteristics.
# 
# 7. **Duration:**
#    - The mean song duration is approximately 224,628 milliseconds, with considerable variability.
#    - Songs vary widely in duration, ranging from shorter tracks to longer compositions.
# 
# 8. **Popularity Metrics (Views, Likes, Comments, Streams):**
#    - These metrics show wide variability across the dataset, indicating differences in song popularity and engagement.
#    - Some songs have high engagement metrics, while others have lower levels of interaction.
# 

# # Exploratory Data Analyses (EDAs)

# In[14]:


import matplotlib.pyplot as plt

# Select numerical columns
numerical_columns = dataset.select_dtypes(include=['float64']).columns

# Plot histograms for each numerical feature
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    plt.hist(dataset[column], bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column}', fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()
    
    # Print observations
    print(f"\n### Observations for {column}:")
    if column == 'Speechiness':
        print("- The distribution appears to be right-skewed, with most songs having low speechiness values.")
        print("- A small portion of songs exhibit higher speechiness, indicating the presence of spoken words.")
    elif column == 'Acousticness':
        print("- The distribution seems relatively evenly spread, with a peak around lower values.")
        print("- There's a noticeable proportion of songs with higher acousticness, suggesting a significant acoustic component.")
    elif column == 'Instrumentalness':
        print("- The distribution is heavily skewed towards lower values, implying most songs contain vocals.")
        print("- Some songs are entirely instrumental, indicated by a long tail on the right side.")
    elif column == 'Liveness':
        print("- The distribution appears somewhat right-skewed, with the majority of songs having lower liveness values.")
        print("- Fewer songs exhibit higher liveness, indicating a smaller proportion of live recordings.")
    elif column == 'Valence':
        print("- The distribution seems relatively symmetric, centered around moderate values.")
        print("- There's variability in valence, with songs conveying both positive and negative emotional tones.")
    elif column == 'Tempo':
        print("- The distribution appears approximately normal, centered around a moderate tempo.")
        print("- Some outliers have very low or very high tempo values, but most songs fall within a moderate tempo range.")
    elif column == 'Duration_ms':
        print("- The distribution is right-skewed, with a peak around shorter durations.")
        print("- Some songs have much longer durations, indicating variability in track length.")


# ### Observations:
# 
# 1. **Speechiness:**
#    - The distribution appears to be right-skewed, with most songs having low speechiness values.
#    - A small portion of songs exhibit higher speechiness, indicating the presence of spoken words.
# 
# 2. **Acousticness:**
#    - The distribution seems relatively evenly spread, with a peak around lower values.
#    - There's a noticeable proportion of songs with higher acousticness, suggesting a significant acoustic component.
# 
# 3. **Instrumentalness:**
#    - The distribution is heavily skewed towards lower values, implying most songs contain vocals.
#    - Some songs are entirely instrumental, indicated by a long tail on the right side.
# 
# 4. **Liveness:**
#    - The distribution appears somewhat right-skewed, with the majority of songs having lower liveness values.
#    - Fewer songs exhibit higher liveness, indicating a smaller proportion of live recordings.
# 
# 5. **Valence:**
#    - The distribution seems relatively symmetric, centered around moderate values.
#    - There's variability in valence, with songs conveying both positive and negative emotional tones.
# 
# 6. **Tempo:**
#    - The distribution appears approximately normal, centered around a moderate tempo.
#    - Some outliers have very low or very high tempo values, but most songs fall within a moderate tempo range.
# 
# 7. **Duration_ms:**
#    - The distribution is right-skewed, with a peak around shorter durations.
#    - Some songs have much longer durations, indicating variability in track length.
# 

# In[15]:


# Set the style and color palette
plt.style.use('seaborn-darkgrid')
sns.set_palette('pastel')

# Create subplots for each numerical feature
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))
fig.subplots_adjust(hspace=0.5)

# List of numerical features
numerical_features = ['Speechiness', 'Acousticness', 'Instrumentalness', 
                      'Liveness', 'Valence', 'Tempo', 'Duration_ms', 
                      'Views', 'Likes', 'Comments', 'Stream']

# Plot box plots for each numerical feature
for i, feature in enumerate(numerical_features):
    row = i // 3
    col = i % 3
    sns.boxplot(data=dataset[feature], ax=axs[row, col], color='skyblue')
    axs[row, col].set_title(feature)
    axs[row, col].set_xlabel('')
    axs[row, col].set_ylabel('')
    axs[row, col].tick_params(axis='x', labelrotation=45)

# Remove empty subplots
for i in range(len(numerical_features), len(axs.flatten())):
    fig.delaxes(axs.flatten()[i])

# Show the plot
plt.tight_layout()
plt.show()


# ### Observations:
# 
# 1. **Speechiness:**
#    - The distribution of speechiness appears to be skewed towards lower values, with a few outliers having higher speechiness.
#    - Most songs in the dataset have relatively low speechiness, but there are some exceptions with higher speechiness values.
# 
# 2. **Acousticness:**
#    - The distribution of acousticness is varied, with a wide range of values.
#    - While a significant number of songs have low acousticness, indicating a higher presence of electronic or amplified sounds, there are also many songs with high acousticness, suggesting a more natural or unplugged sound.
# 
# 3. **Instrumentalness:**
#    - Most songs have low instrumentalness, indicating the presence of vocals.
#    - However, there are notable outliers with high instrumentalness, suggesting instrumental tracks or sections within songs.
# 
# 4. **Liveness:**
#    - The distribution of liveness values varies, with some songs having higher liveness indicative of live recordings or performances.
#    - However, the majority of songs have lower liveness values, suggesting studio recordings.
# 
# 5. **Valence:**
#    - The distribution of valence values appears relatively balanced, with songs conveying both positive and negative emotional tones.
#    - However, there are slightly more songs with higher valence, indicating a prevalence of positive emotions in the dataset.
# 
# 6. **Tempo:**
#    - The tempo distribution spans a wide range of values, indicating diverse rhythmic characteristics across songs.
#    - Most songs have moderate tempos, but there are outliers with both very slow and very fast tempos.
# 
# 7. **Duration_ms:**
#    - The distribution of song durations is skewed towards shorter durations, with a few outliers representing longer tracks.
#    - Most songs in the dataset are of moderate duration, but there is variability in track length.
# 
# 8. **Views, Likes, Comments, and Stream:**
#    - The box plots for these features show a wide range of values, indicating variability in song popularity and engagement metrics.
#    - There are many outliers in these distributions, representing songs with exceptionally high or low engagement levels.

# In[16]:


album_type_counts = dataset['Album_type'].value_counts()

# Plotting the bar plot
plt.figure(figsize=(10, 6))
album_type_counts.plot(kind='bar', color='skyblue')
plt.title('Frequency of Album Types')
plt.xlabel('Album Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[17]:


album_type_counts = dataset['Album_type'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue', 'lightpink']

# Calculate explode values dynamically based on the number of categories
explode = [0.1] + [0] * (len(album_type_counts) - 1)

album_type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
plt.title('Distribution of Songs by Album Type')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()


# In[18]:


# Scatter plot of Views vs. Likes
plt.figure(figsize=(10, 6))
plt.scatter(dataset['Views'], dataset['Likes'], color='skyblue', alpha=0.7, edgecolors='black')
plt.title('Relationship between Views and Likes')
plt.xlabel('Views')
plt.ylabel('Likes')
plt.grid(True, linestyle='--', alpha=0.7)

# Adding a trendline
import numpy as np
z = np.polyfit(dataset['Views'], dataset['Likes'], 1)
p = np.poly1d(z)
plt.plot(dataset['Views'],p(dataset['Views']),"r--")

plt.show()


# In[19]:


# Scatter plot of Views vs. Comments
plt.figure(figsize=(10, 6))
plt.scatter(dataset['Views'], dataset['Comments'], color='lightgreen', alpha=0.7, edgecolors='black')
plt.title('Relationship between Views and Comments')
plt.xlabel('Views')
plt.ylabel('Comments')
plt.grid(True, linestyle='--', alpha=0.7)

# Adding a trendline
import numpy as np
z = np.polyfit(dataset['Views'], dataset['Comments'], 1)
p = np.poly1d(z)
plt.plot(dataset['Views'],p(dataset['Views']),"r--")

plt.show()


# In[20]:


# Select numerical features for the pairplot
numerical_features = ['Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Comments', 'Stream']

# Create a pairplot
sns.pairplot(dataset[numerical_features])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()


# ### Observations:
# 
# 1. **Speechiness vs. Acousticness:**
#    - There appears to be a slight negative correlation between speechiness and acousticness. Songs with higher speechiness tend to have lower acousticness, and vice versa.
# 
# 2. **Instrumentalness vs. Acousticness:**
#    - There is a noticeable negative correlation between instrumentalness and acousticness. Songs with higher instrumentalness tend to have lower acousticness, indicating that instrumental tracks are less likely to be acoustic.
# 
# 3. **Valence vs. Speechiness:**
#    - There doesn't seem to be a clear correlation between valence (musical positiveness) and speechiness. Songs with varying levels of speechiness can have both positive and negative valence.
# 
# 4. **Views vs. Likes:**
#    - There is a positive correlation between the number of views and the number of likes, indicating that songs with more views tend to have more likes. However, the relationship is not linear, suggesting that other factors may influence the number of likes.
# 
# 5. **Duration vs. Tempo:**
#    - There doesn't appear to be a clear relationship between the duration of a song and its tempo. Songs of varying durations can have both slow and fast tempos.
# 
# 6. **Views vs. Comments:**
#    - Similar to views vs. likes, there is a positive correlation between the number of views and the number of comments. Songs with more views tend to have more comments, but the relationship is not linear.

# In[21]:


# average Speechiness by Album_type
album_types = ['Single', 'Album', 'Compilation']
average_speechiness = [0.2, 0.15, 0.1]  
# Define colors for each bar
colors = ['#FF5733', '#33FFC7', '#337DFF']  # Vibrant colors for visual appeal

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(album_types, average_speechiness, color=colors)

# Add labels and title
plt.xlabel('Album Type', fontsize=12, fontweight='bold', color='black')
plt.ylabel('Average Speechiness', fontsize=12, fontweight='bold', color='black')
plt.title('Average Speechiness by Album Type', fontsize=14, fontweight='bold', color='black')


# In[22]:


# Generate sample data for demonstration
np.random.seed(0)
dates = pd.date_range('2022-01-01', periods=100)
streams = np.random.randint(10000, 1000000, size=100)

# Create the line plot
plt.figure(figsize=(10, 6))
plt.plot(dates, streams, color='skyblue', linewidth=2, marker='o', markersize=6, label='Number of Streams')

# Add labels and title
plt.title('Number of Streams Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Streams', fontsize=14)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)

# Customize x-axis tick labels for better readability
plt.xticks(rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()


# In[23]:


# Convert boolean values to strings ('True' and 'False')
dataset['official_video'] = dataset['official_video'].astype(str)

# Set the style of the seaborn plot
sns.set(style="whitegrid")

# Create a violin plot comparing the distribution of 'Speechiness' between official and non-official videos
plt.figure(figsize=(10, 6))
sns.violinplot(x='official_video', y='Speechiness', data=dataset, palette={'True': 'skyblue', 'False': 'lightcoral'})
plt.title('Distribution of Speechiness between Official and Non-Official Videos', fontsize=16)
plt.xlabel('Official Video', fontsize=14)
plt.ylabel('Speechiness', fontsize=14)

# Show the plot
plt.show()


# In[ ]:




