![car-price-tag](https://github.com/footkol/Portfolio/assets/79214748/f69058b8-8188-41ab-b030-22c759d881a5)


In this project, we explore the fundamentals of machine learning using the k-nearest neighbors (KNN) algorithm for univariate and multivariate models. We will apply hyperparameter optimization for each model. In addition, we will also apply various cross-validation methods to assess the performance of our machine learning model.

For the purpose if this project we will first apply 50/50 train/test holdout validation which is actually a specific example (where k = 2) of a larger class of validation techniques called k-fold cross-validation. Following that we will apply k-fold cross validation where k = a range between 3 and 23.

The 50/50 holdout validation has the drawback of utilizing only half of the available data for training. However, using it can help prevent bias towards a specific subset of the data. Holdout validation K-fold cross validation, on the other hand, takes advantage of a larger proportion of the data during training while still rotating through different subsets of the data to avoid the issues of train/test validation.

It's important to note that the choice of a particular validation method depends on the context and the specific requirements of the analysis. While holdout validation allows for an initial assessment of a model's performance, more advanced techniques, such as cross-validation, are often employed to maximize the use of the available data while ensuring an unbiased evaluation of the model's performance. These methods help strike a balance between using the data effectively for training and evaluating the model's generalization capabilities on unseen data.

The project will be carried out in the following sequence of steps:

- Introducing the Dataset
- Data preparation & Normalization
- Univariate k-nearest neighbors model
- Multivariate k-nearest neighbors model
- Adjusting the model for k-fold cross validation
- Conclusion
