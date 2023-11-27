## Data structure

### Traffic sample features

The data was collected from [NYC Open Data](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt)
. The New York City Department of Transportation (NYC DOT) employs Automated Traffic Recorders (ATR) to gather traffic sample volume counts at bridge crossings and roadways. The dataset comprising over 27 million data samples. 

For this exercise, the selected features included date stamps indicating when the count was conducted. These date stamps were separated into distinct columns for the year, month, day, hour, and minute, with data collected at 15-minute intervals. Additional features encompassed the 'On Street' and 'To Street,' providing approximate locations of the Automated Traffic Recorders. The 'Direction' feature represented the text-based direction of traffic at the counting location. The target variable 'Vol' represented the total count collected within 15-minute increments.

In the model development, supplementary features were incorporated, such as weather features and public holiday indicators.

### Weather features

Weather features were obtained from The Meteostat Python library. Meteostat is an open platform which provides free access to historical weather and climate data.
Features included: 
- ‘temp’	The air temperature in °C
- ‘rhum’	The relative humidity in percent (%)
- ‘prcp’	The one hour precipitation total in mm
- ‘wspd’	The average wind speed in km/h
- ‘coco’	The weather condition code
  
The weather condition code, which is hourly weather data, may include information on the observed weather condition. 
Weather conditions are indicated by an integer value between 1 and 27 according to the list below:

- 1	  Clear
- 2	  Fair
- 3  	Cloudy
- 4	  Overcast
- 5	Fog
- 6	Freezing Fog
- 7	Light Rain
- 8	Rain
- 9	Heavy Rain
- 10	Freezing Rain
- 11	Heavy Freezing Rain
- 12	Sleet
- 13	Heavy Sleet
- 14	Light Snowfall
- 15	Snowfall
- 16	Heavy Snowfall
- 17	Rain Shower
- 18	Heavy Rain Shower
- 19	Sleet Shower
- 20	Heavy Sleet Shower
- 21	Snow Shower
- 22	Heavy Snow Shower
- 23	Lightning
- 24	Hail
- 25	Thunderstorm
- 26	Heavy Thunderstorm
- 27	Storm

### Public Holiday features

Public holidays dataset was obtained from Kaggle’s [US Holiday Dates (2004 - 2021)](https://www.kaggle.com/datasets/donnetew/us-holiday-dates-2004-2021)
This list of holidays includes 18 years of US Holidays dated between January 1st, 2004 and December 31st, 2021. Each record has Date, Holiday, Weekday, Month, Day and Year.

Included Holidays:
- 4th of July
- Christmas Eve & Christmas Day
- Columbus Day
- Eastern & Western Easter
- Juneteenth
- Labor Day & Labor Day Weekend
- Martin Luther King, Jr. Day
- Memorial Day
- New Year’s Eve & New Year's Day
- Thanksgiving Eve & Thanksgiving Day
- Valentine’s Day
- Veterans Day
- Washington's Birthday


## Model selection


While selecting a model for this project I decided to opt for Neural Networks approach.  Selecting deep learning models, is advantageous for traffic prediction due to its inherent ability to capture intricate patterns, model complex relationships and adapt to varying temporal dependencies. Unlike traditional machine learning models, neural networks excel at automatically learning hierarchical representations, enabling them to extract relevant features from diverse data types, such as time-series and categorical information. This adaptability, coupled with the capacity to handle non-linearities, makes neural networks a superior choice for traffic prediction tasks compared to conventional machine learning models.

Recurrent Neural Networks (RNNs) are a type of neural network designed for handling sequential data. What sets RNNs apart is their ability to retain information from previous inputs, making them effective for tasks where context or order matters, such as time-series prediction. However, traditional RNNs can face challenges with long-term dependencies, leading to the development of more advanced architectures like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, which address these issues by allowing the model to selectively retain or forget information over time.

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are both types of recurrent neural network (RNN) architectures designed to address the vanishing gradient problem in traditional RNNs, allowing them to capture long-term dependencies in sequential data.
