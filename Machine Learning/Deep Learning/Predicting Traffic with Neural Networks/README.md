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

There are several well-researched methods applicable to creating traffic prediction models. First is the use of statistical methods, allowing the identification of traffic patterns at different scales: daily, on different days of the week, seasonal, etc. These methods are usually easier, faster, and more cost-effective to implement compared to machine learning approaches. However, they tend to be less accurate since they cannot process as much multivariate data.

In particular, auto-regressive integrated moving average (ARIMA) models have been actively employed for traffic prediction since the 1970s. They are easy to implement and have demonstrated higher accuracy compared to other statistical methods. ARIMA follows a classical statistical approach, analyzing past events to predict future ones. It relies on data collected at regular time intervals and assumes that past patterns will repeat in the future. However, traffic flow is a complex structure with numerous variables, making it challenging to effectively process using univariate ARIMA models.

On the other hand, machine learning offers the advantage of not requiring assumptions or prior knowledge. It can automatically extract useful information from datasets, compensating for the limitations of traditional methods. Ensemble learning, a significant branch of machine learning, has gained widespread attention. It combines multiple learners to reduce generalization errors, minimize the possibility of local optimization, and address overfitting.

For this project, I chose LightGBM, a powerful open-source gradient boosting framework designed for efficiency and high performance. It proves to be an excellent tool for handling large datasets and facilitating the creation of accurate predictions.

In addition to selecting the model based on decision tree algorithms mentioned above, I have also opted for a Neural Networks approach.  Selecting deep learning models, is advantageous for traffic prediction due to its inherent ability to capture intricate patterns, model complex relationships and adapt to varying temporal dependencies. Unlike traditional machine learning models, neural networks excel at automatically learning hierarchical representations, enabling them to extract relevant features from diverse data types, such as time-series and categorical information. This adaptability, coupled with the capacity to handle non-linearities, makes neural networks a superior choice for traffic prediction tasks compared to conventional machine learning models.

Recurrent Neural Networks (RNNs) are a type of neural network designed for handling sequential data. What sets RNNs apart is their ability to retain information from previous inputs, making them effective for tasks where context or order matters, such as time-series prediction. However, traditional RNNs can face challenges with long-term dependencies, leading to the development of more advanced architectures like Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, which address these issues by allowing the model to selectively retain or forget information over time.

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are both types of recurrent neural network (RNN) architectures designed to address the vanishing gradient problem in traditional RNNs, allowing them to capture long-term dependencies in sequential data.

## Final results


The charts showcase the predicted values from all four models alongside the actual values for easy comparison.
### LightGBM
![LightGBM](https://github.com/footkol/Portfolio/assets/79214748/b62689c0-954e-49f0-9c78-7c46711006cc)

### LSTM
![LSTM](https://github.com/footkol/Portfolio/assets/79214748/01888003-30d2-422b-aec7-640b35f4ac34)

### GRU
![GRU](https://github.com/footkol/Portfolio/assets/79214748/4283bbb0-392c-4e6b-a2d2-3b89de49aff2)

### RNN
![RNN](https://github.com/footkol/Portfolio/assets/79214748/a9261e9c-2991-43b1-beff-d7dc36d6249a)

LightGBM model had displayed the best performance in comparison to neural network models. Among the RNN models LSTM showed the best results. 

![results](https://github.com/footkol/Portfolio/assets/79214748/cbd55df3-c2be-4f5d-bec2-ba0befc7b528)


Lower MSE, MAE and RMSE indicate better model performance. The R2 score ranges from 0 to 1. A score of 1 indicates that the model perfectly predicts the dependent variable, while a score of 0 suggests that the model provides no improvement over a simple mean-based model.

However, it is essential to remember that these interpretations are general guidelines, and the context of our specific problem and the characteristics of our data should be considered when evaluating model performance. It's important to consider these metrics collectively to gain a comprehensive understanding of the models' performance.

Visual comparison of LSTM with LightGBM model
![LSTM and LightGBM](https://github.com/footkol/Portfolio/assets/79214748/4ca7c546-0181-463f-acea-0050149acc72)

## Next steps

The model has room for improvement, and depending on the available data, enhancements can be achieved by adjusting various hyperparameters such as the number of layers, units, epochs, or learning rate to optimize performance.

Once you are content with your model, deployment for predictions on new or unseen data is possible. Utilizing tools like TensorFlow Serving, PyTorch Serve, or Flask, you can establish a web service or API capable of receiving and responding to requests from clients or applications. Hosting and scaling the model in the cloud can be accomplished through platforms like AWS, Google Cloud, or Azure.

Maintenance of the model is crucial over time, considering potential changes in data and the environment. Employing strategies such as retraining, updating, or fine-tuning ensures the model stays current with the latest data and trends. Techniques like anomaly detection, drift detection, or feedback loops can be applied to monitor and identify changes or issues in the model's performance or behavior.

## Note

GitHub performs a static render of the notebooks and it doesn't include the embedded HTML/JavaScript that makes up a plotly graph. In order to see plotly graphs in a rich view of the notebook please paste the link to this GitHub notebook into http://nbviewer.jupyter.org/ 
