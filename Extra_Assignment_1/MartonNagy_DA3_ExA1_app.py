import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('simulated_data.csv')
    
def plot_bias_variance_tradeoff(results):
    # Fig 1. plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # First y-axis for Bias²
    ax1.plot(results['degree'], results['bias'], label='Bias²', color='skyblue', marker='o')
    ax1.set_xlabel('Model complexity (degree)', fontsize=12)
    ax1.set_ylabel('Bias²', color='skyblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.grid(visible=True)

    # Second y-axis for Variance
    ax2 = ax1.twinx()
    ax2.plot(results['degree'], results['variance'], label='Variance', color='orange', marker='o')
    ax2.set_ylabel('Variance', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    st.pyplot(plt.gcf())

def plot_bias_variance_pie(mnum, bias_squared, variance, ax):
    # Fig. 2 plot
    labels = ['Bias²', 'Variance']
    sizes = [bias_squared, variance]
    colors = ['skyblue', 'orange']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.set_title(f'MSE breakdown of Model {mnum}', fontsize=14)

def main():
    st.title("MSE bias-variance decomposition illustration on a diamond price prediction problem")
    st.write("Author: Marton Nagy")
    st.write("Prepared as an extra assignment for Data Analysis 3 at CEU.")

    # Sidebar set-up
    st.sidebar.header("Settings")

    # Figure 1 inputs
    st.sidebar.subheader('Inputs for Fig. 1')
    max_degree = st.sidebar.slider("Maximum polynomial degree", 1, 20, 10, step = 1)
    #nruns = st.sidebar.slider("Number of iterations", 1, 200, 50, step = 1)
    x_vars = st.sidebar.multiselect("Predictors to include in models", options = ['carat', 'table', 'depth'], default = 'carat')
    # Fallback to default for when no vars are selected.
    if len(x_vars) == 0:
        st.sidebar.write('Please select at least one variable! If none are selected, carats are shown by default!')
        x_vars = ['carat']
    
    # Figure 2 inputs
    st.sidebar.subheader('Inputs for Fig. 2')
    m1_degree = st.sidebar.slider("Polynomial degree of Model 1", 1, 20, 2, step = 1)
    m1_vars = st.sidebar.multiselect("Predictors in Model 1", options = ['carat', 'table', 'depth'], default = 'carat')
    if len(m1_vars) == 0:
        st.sidebar.write('Please select at least one variable! If none are selected, carats are shown by default!')
        m1_vars = ['carat']
    m2_degree = st.sidebar.slider("Polynomial degree of Model 2", 1, 20, 10, step = 1)
    m2_vars = st.sidebar.multiselect("Predictors in Model 2", options = ['carat', 'table', 'depth'], default = ['carat', 'table', 'depth'])
    if len(m2_vars) == 0:
        st.sidebar.write('Please select at least one variable! If none are selected, carats are shown by default!')
        m2_vars = ['carat']
    
    #Introduction
    st.subheader("Introduction")
    st.write("""As discussed in class, the Mean Squared Error may be decomposed into two components: bias (squared) and variance. The bias of a prediction 
    means that how much we are off in our predictions, on average. The variance captures how the prediction error varies around its mean across different 
    predictions. Generally speaking, there is a trade-off between bias and variance of a prediction. The purpose of this application is to 
    illustrate this problem on a real-life prediction problem.""")
    
    # Data description
    st.subheader("Dataset description")
    st.write("""This app uses the well-known diamonds dataset. This classic dataset contains physical attributes and prices of 53,940 diamonds.
    The target variable of the model is price in US dollars. Features you can select into the models are (1) carat: weight of the diamond; 
    (2) depth: total depth percentage (calculated as 100 * depth / mean(width, height)); and (3) table: width of the top of the diamond relative 
    to the widest point. You can also select the maximum number of polynomial degrees to be shown on the chart.""")
    
    # Figure 1 computation and plotting
    st.subheader("Fig. 1: Visualizing the bias-variance trade-off")
    results = (data[
               (data['carat'] == int('carat' in x_vars))
               & (data['depth'] == int('depth' in x_vars))
               & (data['table'] == int('table' in x_vars))
               & (data['degree'] <= max_degree)
    ])
    plot_bias_variance_tradeoff(results)

    st.subheader("Interpretation")
    st.write("""What we can see from the chart is that more simplistic models tend to produce lower variance but higher bias.
    On the other hand, overly complex models show higher variance but lower bias. So selecting the best model for prediction is indeed a balancing game: 
    we want to aim for the sweet spot somewhere in the middle where both bias and variance are relatively low. Note, however, that where this spot lies 
    depends not only on the polynomial degrees of the model (one kind of complexity), but also on how many variables we include. If we play around with 
    the inputs a bit, we can see that for models containing more and more variables, the sweet spot in terms of polynomial degrees is lower and lower. Also, 
    after surpassing the balanced point, we might observe some interesting things: even though the variance part tends to grow with model complexity 
    (as expected), the bias component shows some interesting variation between very complex models.""")

    # Figure 2
    st.subheader("Comparing two models directly")
    st.write("""You can also compare two models of different complexity. For this, please use the relevant inputs on the sidebar.
    With this visualization, you can directly see how much models of different complexities differ in their MSE decompositions.""")
    st.subheader("Fig. 2: Comparison of two regression models' MSE decomposition")
    
    # Figure 2 plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    bias_squared_1 = (data[
               (data['carat'] == int('carat' in m1_vars))
               & (data['depth'] == int('depth' in m1_vars))
               & (data['table'] == int('table' in m1_vars))
               & (data['degree'] == m1_degree)
    ]).bias.iloc[0]
    variance_1 = (data[
               (data['carat'] == int('carat' in m1_vars))
               & (data['depth'] == int('depth' in m1_vars))
               & (data['table'] == int('table' in m1_vars))
               & (data['degree'] == m1_degree)
    ]).variance.iloc[0]
    bias_squared_2 = (data[
               (data['carat'] == int('carat' in m2_vars))
               & (data['depth'] == int('depth' in m2_vars))
               & (data['table'] == int('table' in m2_vars))
               & (data['degree'] == m2_degree)
    ]).bias.iloc[0]
    variance_2 = (data[
               (data['carat'] == int('carat' in m2_vars))
               & (data['depth'] == int('depth' in m2_vars))
               & (data['table'] == int('table' in m2_vars))
               & (data['degree'] == m2_degree)
    ]).variance.iloc[0]
    plot_bias_variance_pie(1, bias_squared_1, variance_1, axes[0])
    plot_bias_variance_pie(2, bias_squared_2, variance_2, axes[1])
    st.pyplot(fig)
    st.write("""This visualization also sheds some light on something that may not have been straightforward from Fig. 1 
    (because of the differently scaled axes): the optimal spot of complexity usually does not mean that the two components of MSE are equal to each other. 
    In this example, e.g., the squared bias is usually (much) larger than the variance - even for the more complex models.""")

    # Tech notes
    st.subheader("Some technical notes")
    st.write("""The way this dashboard calculates the MSE decomposition is quite complex. First, it creates a 2:1 train-test split randomly.
    Then, using the mlxtend package, it creates bootstrap sample from the train set a number of times set by the user to mimic variability in the 
    original data. Then, the models estimated over the bootstrap samples are evaluated over the test set. Finally, the needed metrics are 
    calculated by using all the bootstrap predictions over the test sample. Note that every model estimation have been done beforehand and the results are 
    stored in a CSV file for dashboard efficiency.""")

    # AI use notes
    st.subheader("AI use disclaimer")
    st.write("""During this project, generative AI (ChatGPT-4o) has been used for the following purposes: 
    (1) selecting an appropriate dataset, (2) Streamlit app skeleton and troubleshooting, and (3) Matplotlib code skeletons.""")

if __name__ == "__main__":
    main()
