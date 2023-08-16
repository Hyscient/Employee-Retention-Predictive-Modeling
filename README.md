# Employee-Retention-Predictive-Modeling

## Overview
The objective of this project was to develop predictive models using logistic regression, decision tree, and random forest algorithms to identify factors contributing to employee departures in Salifort Motors. The analysis focused on evaluating employee retention patterns and creating models that can assist in identifying employees who are more likely to leave. The final random forest model achieved an accuracy of 96.1% and an AUC score of 93.8%, showcasing its ability to effectively predict employee departures.

## Business Understanding
Employee retention is a critical concern for organizations aiming to maintain a stable and productive workforce. High employee turnover can lead to increased costs, decreased morale, and potential disruptions in operations. Identifying key factors influencing employee departures can enable proactive strategies to enhance employee engagement and reduce turnover rates.

## Data Understanding
The dataset used for this analysis contained information from Salifort Motors' HR department, encompassing various features such as employee tenure, performance evaluations, project involvement, work accidents, salary levels, and department affiliations. The dataset was preprocessed to handle missing values, convert categorical variables, and engineer relevant features. A breakdown of the employee turnover distribution revealed insights into the data.

## Modeling and Evaluation
Three different machine learning models were developed to predict employee departures: logistic regression, decision tree, and random forest. The random forest model, consisting of an ensemble of decision trees, outperformed the other models with an accuracy of 96.1% and an AUC score of 93.8%. The feature importance analysis highlighted critical factors contributing to employee departures, including last evaluation scores, number of projects, tenure, and workload perception.

## Conclusion
The developed random forest model offers a powerful tool for identifying employees at risk of departure, enabling Salifort Motors to take proactive measures to retain valuable talent. However, further exploration can delve into predicting the extent of employee satisfaction and the potential influence of additional features on the model's predictions. This analysis provides actionable insights for improving employee retention strategies, fostering a positive work environment, and ultimately contributing to Salifort Motors' success.

## Next Steps
1. Investigate potential data leakage and assess model performance without the `last_evaluation` feature.
2. Explore the use of K-means clustering to gain deeper insights into underlying employee engagement patterns.
3. Conduct a parametric analysis to quantify the impact of each variable on the likelihood of employee departure.
4. Consider incorporating historical employee engagement data to refine the model and address business needs effectively.

## Resources Used
- Python: NumPy, pandas, scikit-learn, Matplotlib, Seaborn
- Jupyter Notebook
- HR data from Salifort Motors

## Ethical Considerations
Throughout the analysis, ethical considerations were upheld to ensure responsible data handling and interpretation. Privacy and data protection were maintained in accordance with relevant regulations, and the focus remained on utilizing the data to generate valuable insights for the benefit of the organization and its employees.
