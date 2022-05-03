import sys
sys.path.append('C:/Users/97254/PycharmProjects/IML.HUJI')
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import multivariate_normal
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    univariate = UnivariateGaussian()
    univariate.fit(samples)
    print("Q1 - estimated expectation and variance:")
    print((univariate.mu_, univariate.var_))

    # Question 2 - Empirically showing sample mean is consistent
    samples_num = 10
    distance = []
    while samples_num <= 1000:
        samples_vec = np.random.normal(10, 1, samples_num)
        univariate.fit(samples_vec)
        distance.append(np.abs(10 - univariate.mu_))
        samples_num += 10
    samples_num = np.linspace(10, 1000, 100)

    go.Figure([go.Scatter(x=samples_num, y=distance, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{Q2 - Absolute distance between the estimated "
                        r"and true value of the expectation as a function of the sample size}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="absolute distance from real expectation",
                  height=600)).show()
    # Question 3 - Plotting Empirical PDF of fitted model
    samples.sort()
    pdf = univariate.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=pdf, mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title="Q3 - Empirical pdf function under the fitted model",
                  xaxis_title="Samples",
                  yaxis_title="pdf",
                  height=500)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.transpose(np.array([0, 0, 4, 0]))
    sigma = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    multivariate = MultivariateGaussian()
    multivariate.fit(samples)
    print("Q4 - estimated and covariance:")
    print(multivariate.mu_)
    print(multivariate.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = f1
    values = []
    f1_max, f3_max = -10, -10
    max_likelihood = -np.inf
    for mu1 in f1:
        for mu3 in f3:
            values.append(multivariate.log_likelihood(
                np.transpose(np.array([mu1, 0, mu3, 0])), sigma, samples))
            if values[-1] > max_likelihood:
                max_likelihood = values[-1]
                f1_max, f3_max = mu1, mu3
    values = np.array(values).reshape(200, 200)
    go.Figure(go.Heatmap(x=f1, y=f3, z=values),
              layout=go.Layout(title="Q5 - log-likelihood - heatmap",
                               height=500, width=500)).show()

    # Question 6 - Maximum likelihood
    print(
        "Q6 - (f1_max, f3_max) : " + str((round(f1_max, 3), round(f3_max, 3))))



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
  
