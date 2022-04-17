from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px



def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu_, var_, = 10, 1
    estimation = UnivariateGaussian()
    X = np.random.normal(mu_,var_,1000)
    estimation.fit(X) #notes I avoid reassignment (  ans = ans.fit(X))
    print((estimation.mu_, estimation.var_))


    # Question 2 - Empirically showing sample mean is consistent

    samples = np.arange(10, 1010, 10)  #arange allow you to define the size of the step. linspace allow you to define the number of steps. –
    abs_dist = []
    for sample in samples:
        estimation1 = UnivariateGaussian().fit(X[:sample]).mu_
        abs_dist.append(abs(estimation1 - mu_))

    go.Figure([go.Scatter(x=samples, y=abs_dist, mode='markers+lines',
                          name=r'$\widehat\Q2$')],
                 layout=go.Layout(
                  title=r"$\text{Difference in expectation "
                        r"Of Number Of Samples}$",
                  xaxis_title="$\\text{Num of samples}$",
                  yaxis_title="$\\text{difference in estimation}$",
                  height=400)).show()


    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure([go.Scatter(x=X, y=estimation.pdf(X), mode='markers',
                          name=r'$\widehat\mu$')],
                layout=go.Layout(
                  title=r"$\text{ PDFS of samples }$",
                  xaxis_title="$\\text{ SAMPLES Number}$",
                  yaxis_title="PDF",
                  height=600)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu_ = np.array([0, 0, 4, 0])
    cov_ = np.array([[1,0.2,0,0.5],[0.2,2,0,0],[0,0,1,0],[0.5,0,0,1]])
    X = np.transpose(np.random.multivariate_normal(mu_, cov_, size=1000)) #TODO: why do I need transpose here (only seems to work)
    estimation = MultivariateGaussian()
    estimation.fit(X)

    print("mu: \n " + str(estimation.mu_))
    print("Cov matrix: \n " + str(estimation.cov_))


    # Question 5 - Likelihood evaluation

    f3s = np.linspace(-10, 10, 200)
    f1s = np.linspace(-10, 10, 200)
    estimation = []
    mu_s = []
    for f1 in f1s:
        for f3 in f3s:
            mu_ = np.array([f1, 0, f3, 0])
            mu_s.append(mu_)
            estimation.append(MultivariateGaussian().log_likelihood(mu_, cov_, X))
    mu_s = np.asarray(mu_s)
    estimation = np.array(estimation).reshape((200, 200))
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=f1s, y=f3s,
                             z=estimation,
                             colorbar=dict(title="LogLikelihood")))
    fig.update_layout(xaxis_title="$\\text{f1s values}$",
                      yaxis_title="$\\text{f3s values}$",
                      height=600)
    fig.show()

    # Question 6 - Maximum likelihood
    
    best_model = mu_s[np.argmax(estimation)]
    f1,f3 = best_model[0], best_model[2]
    print('max_pair of values from f1s and f3s respectively\n')
    print(round(f1,3))
    print(round(f3, 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()





