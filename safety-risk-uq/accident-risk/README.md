# Aviation Risk Estimation
## Introduction
In this module, we are trying to simulate aviation accident through NATS and quantify the evolving risk during the accident. More specifically, there are two major contribitions in this module. First of all, we convert the aviation accident recordings from NTSB into data that can be identified through NATS.  Secondly, we build up a risk estimation pipline using deep learning and tree-based method according to the NTSB data. Once the module is able to accurately capture enough details in the aviation accident, more scientific research about aviation accident can be done according to the module then. 
## Requirement
- Pytorch 1.4.0
- catboost 0.23
- xgboost 1.0.2 
## Module details
- `simulation_method/aviationr.py` is the major module which follows the template format for NATS simulation
  - `class AviationRisk` is the major class 
    - function `accident_simulator` provides the interface between NTSB and NATS 
- `simulation_method/aviationr_model.py` is providing necessary classes for pre-trained model and the class for risk estimation 
  - `class RNNModel` provides the architecture of hieachical LSTM
  - `class SequentialPrediction` provides the architecture of hieachical markovian
  - `class HierarchicalSoftmax` provides the architecture of hieachical softmax
  - `class RiskEstimator` provides the pipline for estimating the risk for each event in aviation accident
- `tests/aviationr_example.py` provides a simple illustration example
- `sample_data/aviationR/` contains the required data for the example
  - `sample_data/aviationR/data` includes the necessary information for the aviation accident
  - `sample_data/aviationR/model` includes the pre-trained models
- `setup.py` shows three extra packages that needed in in module
  - pytorch 1.4.0
  - catboost 0.23
  - xgboost 1.0.2
## Methodology
### Accident Simulator

Accident simulator is for exploring what and how we build NSTB accident recordings into NATS simulator. According to our current understanding about NATS. The following interface will cooperate to simulate the aviaion accident

- [x] aircraftInterface
- [ ] environmentInterface
- [ ] controllerInterface
- [ ] pilotInterface

Currently, we only implement a simple simulator through setting the phase of flight. We find the relationship of definition between NATS and NTSB manually. And the table needs to be improved furthermore with more knowledge. Following is an example of current table. 

|      | NTSB_CODE | NTSB                          | NATS_CODE | NATS                     |
| ---- | --------- | ----------------------------- | --------- | ------------------------ |
| 0    | 500       | Standing                      | 1.0       | FLIGHT_PHASE_ORIGIN_GATE |
| 1    | 501       | Standing - pre-flight         | 1.0       | FLIGHT_PHASE_ORIGIN_GATE |
| 2    | 502       | Standing - starting engine(s) | 1.0       | FLIGHT_PHASE_ORIGIN_GATE |

### Risk Estimator

We further implement a risk estimation module based on deep learning and tree-based method. We build a hierchical LSTM as a sequential model to simulate the sequential events in an aviation accident. Each sequential events will be represented using one hot encoding and be classified into minor/ substantial/ destroyed which is also defined according to NTSB. The risk estimator modules contains the following two major modules 

#### Future Event Predictoin and Hierarhical Tree-based Embedding: 

We propose a tree-based embedding method to map the original hierarhical event representation into a low-dimensional vectors. These embedding will be used in an LSTM to predict the future events. The models are saved in sample_data/aviationR.  and the code are saved in simulation_method/aviationr_model.py. Here, `Pytorch` is required since deep learning modules implemented in the proposed framework. 

#### Risk Event Quantification

Predict the future risk and give its uncertainty. To achieve this, Both Xgboost and Catboost will be used combining with the future event prediction module. 
