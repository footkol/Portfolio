## Introduction

The project focused on predicting traffic in New York City. To build the model, a specific location with the highest volume of time stamps was selected for training, emphasizing a concentrated dataset for enhanced model accuracy.

## Data structure

### Traffic sample features

The data was collected from [NYC Open Data](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt)
. The New York City Department of Transportation (NYC DOT) employs Automated Traffic Recorders (ATR) to gather traffic sample volume counts at bridge crossings and roadways. The dataset comprising over 27 million data samples. 

For this exercise, the selected features included date stamps indicating when the count was conducted. These date stamps were separated into distinct columns for the year, month, day, hour, and minute, with data collected at 15-minute intervals. Additional features encompassed the 'On Street', 'From Street' and 'To Street,' providing more precise locations of the Automated Traffic Recorders. The 'Direction' feature represented the text-based direction of traffic at the counting location. The target variable 'Vol' represented the total count collected within 15-minute increments.

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

## Initial Data Preprocessing 

The attempt was made to identify a location with the largest amount of historical data. As a result, Flatbush Avenue in Brooklyn, one of the five administrative divisions of New York City, was chosen. Specifically, the portion between the 'From Street' (Brighton Line) and 'To Street' (Brighton Line) features exhibited the highest number of time stamps between 2012 and 2019. The BMT Brighton Line, also known as the Brighton Beach Line, is a rapid transit line in the B Division of the New York City Subway in Brooklyn, runs parallel to Flatbush Avenue. 

![image](https://github.com/footkol/Portfolio/assets/79214748/2f19babf-c8df-44e5-bd16-26f353b26dcc)


## Model selection


While selecting a model for this project I decided to opt for Neural Networks approach.  Selecting deep learning models, is advantageous for traffic prediction due to its inherent ability to capture intricate patterns, model complex relationships and adapt to varying temporal dependencies. Unlike traditional machine learning models, neural networks excel at automatically learning hierarchical representations, enabling them to extract relevant features from diverse data types, such as time-series and categorical information. This adaptability, coupled with the capacity to handle non-linearities, makes neural networks a superior choice for traffic prediction tasks compared to conventional machine learning models.

Recurrent Neural Networks (RNNs) are a type of neural network designed for handling sequential data. What sets RNNs apart is their ability to retain information from previous inputs, making them effective for tasks where context or order matters, such as time-series prediction. However, traditional RNNs can face challenges with long-term dependencies, leading to the development of more advanced architectures like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, which address these issues by allowing the model to selectively retain or forget information over time.

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are both types of recurrent neural network (RNN) architectures designed to address the vanishing gradient problem in traditional RNNs, allowing them to capture long-term dependencies in sequential data.

## Final results


The chart showcases the predicted values from both LSTM and GRU models alongside the actual values for easy comparison.

![newplot](https://github.com/footkol/Portfolio/assets/79214748/0a6d6452-4630-4bf2-aaec-93e5bccdae8e)

The models evaluation was based on comapring Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) and R-squared (R2) values:
- RNN metrics: {'mae': 8.060538, 'rmse': 10.506156932774884, 'r2': 0.43302360364046144}
- GRU metrics: {'mae': 7.6822305, 'rmse': 10.218281119785694, 'r2': 0.4636689643741928}
- LSTM metrics: {'mae': 7.8293996, 'rmse': 10.699591684126608, 'r2': 0.411953573268795}

Lower MAE values indicate better performance. Both GRU and LSTM have lower MAE compared to RNN, suggesting they are better at predicting the target variable.

Similar to MAE, lower RMSE values indicate better performance. In this case, GRU has the lowest RMSE.

R² ranges from 0 to 1, with 1 indicating a perfect fit. Higher R² values suggest that a larger proportion of the variance in the target variable is explained by the model. In this case, GRU has the highest R², indicating better explanatory power compared to RNN and LSTM.

It's important to consider these metrics collectively to gain a comprehensive understanding of the models' performance. 

## Next steps

The model has room for improvement, and depending on the available data, enhancements can be achieved by adjusting various hyperparameters such as the number of layers, units, epochs, or learning rate to optimize performance.

Once you are content with your model, deployment for predictions on new or unseen data is possible. Utilizing tools like TensorFlow Serving, PyTorch Serve, or Flask, you can establish a web service or API capable of receiving and responding to requests from clients or applications. Hosting and scaling the model in the cloud can be accomplished through platforms like AWS, Google Cloud, or Azure.

Maintenance of the model is crucial over time, considering potential changes in data and the environment. Employing strategies such as retraining, updating, or fine-tuning ensures the model stays current with the latest data and trends. Techniques like anomaly detection, drift detection, or feedback loops can be applied to monitor and identify changes or issues in the model's performance or behavior.
