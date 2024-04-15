# JDFlow
### Kirill Zakharov

The official realisation of the prposed method JDFlow (link on the article will be available later).

#### Abstract: 


#### Training hyperparameters
We employ three methods for generating multivariate time series to compare against our approach: Fourier Flow, fractional-SDE and PAR from SDV. We train Fourier Flow for 1000 epochs with 10 flows. The hidden dimension of the model is equal to 256. For the PAR synthesizer we utilise $1000$ epochs and for a metadata we additionally use the time index with uniform range. The fractional SDE-Net model was trained by 1000 epochs and batch dimension 5.
