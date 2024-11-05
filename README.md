# StressSense: Stress Level Detection Through Sleep Quality - MLOps


| | Description |
| ----------- | ----------- |
| Dataset | [Human Stress Detection Through Sleep Quality](https://www.kaggle.com/datasets/laavanya/human-stress-detection-in-and-through-sleep?select=SaYoPillow.csv) |
| Problem Statemet | With today's fast-paced lifestyle, prioritizing sleep quality is often overlooked, which can lead to a range of health issues, even in younger individuals. Stress, in particular, is both a cause and a consequence of poor sleep, leading to a harmful cycle. Rachakonda et al [1]  proposed the Smart-Yoga Pillow, an edge device that analyzes body vitals and securely transfers data to the cloud to explore the connection between stress and sleep. |
| Machine Learning Solution | Using machine learning, this project analyzes stress and sleep quality based on key physiological features, enabling early detection of distress. By identifying stress indicators, preventive actions can be taken, improving users' overall well-being. |
| Data Processing | The dataset contains numerical features with categorical labels. Preprocessing includes label encoding and feature normalization. Given the dataset's relatively small size, an 80/20 train-test split is applied.|
| Model Architecture | The model is a straightforward neural network with an **input** layer,a **hidden** layer consisting of two dense layers with dropout, and an **output** layer. his simple architecture is appropriate due to the dataset already having key features and cleaned, where a lightweight neural network suffices. Hyperparameter tuning is managed with TFX Tuner. |
| Evaluation metrics | AUC, Precision, Recall, Example Count, and Accuracy. |
| Model Performance | During training and tuning, the model achieved 100% accuracy on both training and test sets, demonstrating a high ability to classify stress levels based on sleep quality data. |
| Deployment Option | The service is deployed on [railway](https://railway.app/). |
| Web App | The service is accessible through the [stress-sleep-model-app](https://vigilant-surprise-production.up.railway.app/v1/models/stress_sleep_model/metadata).|
| Monitoring | Prometheus and Grafana are integrated to monitor the deployed service. Currently, key metrics include latency and request counts, using buckets and various aggregeations. This setup serves as an MVP, and future iterations will include more comprehensive monitoring. |

---

In this repository, you can find the request methods through the test-prediction notebook. The stress levels are categorized into 5 levels, from low/normal, medium_low, medium, medium_high, and high/unhealthy. In the real life scenario, medium_high and high categories are strongly advised to seek professional attention. 

The ML pipeline of this service is fully developed under the Tensorflow Extended (TFX) framework. TFX is an end-to-end MLOps environment equipped to build production level ML service. The process of architecturing this project also follows the Clean Code paradigm, promoting modularity for future managements, increments, and maintenance.

*Prometheus Dashboard*
![prometheus dashboard](docs/prometheus-monitoring.png?raw=True)


*Grafana Dashboard*
![grafana dashboard](docs/grafana-dashboard.png?raw=True)

---
References:

[1] L. Rachakonda, A. K. Bapatla, S. P. Mohanty, and E. Kougianos, “SaYoPillow: Blockchain-Integrated Privacy-Assured IoMT Framework for Stress Management Considering Sleeping Habits”, IEEE Transactions on Consumer Electronics (TCE), Vol. 67, No. 1, Feb 2021, pp. 20-29.