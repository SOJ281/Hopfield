# Hopfield

This Github contains multiple hopfield network implementations

They are all contained in the hopfield.py file.

All test scripts can be run using the following command:
    
    python3 <script>


# Documentation:
##hopfield.py
### Hopfield Class:

Implements a basic Hopfield network.

**__init__(inputs):**

Initializes a Hopfield network with given input patterns.

    Parameters:
        inputs (list of arrays): List of input patterns (bipolar).

    Returns:
        None

**predict(input, iterations, theta=0.0):**

Predicts the output pattern given an input pattern.

    Parameters:
        input (array): Input pattern.
        iterations (int): Maximum number of iterations.
        theta (float, optional): Threshold for energy.

    Returns:
        predicted (list of arrays): Predicted patterns at each iteration.

### DAMDiscreteHopfield Class:

Implements a Dense Associative Memory (DAM) Hopfield network for discrete patterns.

**__init__(inputs):**

Initializes a DAM Hopfield network with given input patterns.

    Parameters:
        inputs (list of arrays): List of input patterns (bipolar).

    Returns:
        None

**predict(input, iterations):**

Predicts the output pattern given an input pattern.

    Parameters:
        input (array): Input pattern.
        iterations (int, optional): Maximum number of iterations.

    Returns:
        predicted (list of arrays): Predicted patterns at each iteration.

### ContinuousHopfield Class:

Implements a Hopfield network for continuous patterns and is adjusted to do mean.

**__init__(inputs):**

Initializes a Continuous Hopfield network with given input patterns.

    Parameters:
        inputs (list of arrays): List of input patterns (continuous).

    Returns:
        None

**predict(input, iterations, beta=8):**

Predicts the output pattern given an input pattern.

    Parameters:
        input (array): Input pattern.
        iterations (int, optional): Maximum number of iterations.
        beta (float, optional): Temperature parameter for softmax function.

    Returns:
        predicted (list of arrays): Predicted patterns at each iteration.

### ContinuousHopfield Class:

Implements a Hopfield network for continuous patterns.

**__init__(inputs):**

Initializes a Continuous Hopfield network with given input patterns.

    Parameters:
        inputs (list of arrays): List of input patterns (continuous).

    Returns:
        None

**predict(input, iterations, beta=8):**

Predicts the output pattern given an input pattern.

    Parameters:
        input (array): Input pattern.
        iterations (int, optional): Maximum number of iterations.
        beta (float, optional): Temperature parameter for softmax function.

    Returns:
        predicted (list of arrays): Predicted patterns at each iteration.

### SimplicialHopfield Class:

Implements a Hopfield network using setwise connections.

**__init__(inputs):**

Initializes a Simplicial Hopfield network with given input patterns.

    Parameters:
        inputs (list of arrays): List of input patterns (continuous).

    Returns:
        None

**predict(input, iterations, theta=0.0):**

Predicts the output pattern given an input pattern.

    Parameters:
        input (array): Input pattern.
        iterations (int, optional): Maximum number of iterations.
        theta (float, optional): Threshold for energy.

    Returns:
        predicted (list of arrays): Predicted patterns at each iteration.

##imageTest.py

This script implements several functions for image processing, image corruption, and visualization. It uses the Hopfield network implementations from the hopfield module to perform predictions on corrupted images.

###Functions:

**randomFlipping(input, flipCount):**
Randomly inverts values in an input array based on a specified flip probability.

    Parameters:
        input (array): The input data.
        flipCount (float): The probability of flipping a value.

    Returns:
        flippy (array): A copy of the input array with random flips.

**randomBlocking(input, blockLevel):**
Blocks a random chunk of data from the input based on the specified block level.

    Parameters:
        input (array): The input data.
        blockLevel (float): The proportion of data to block.

    Returns:
        blocked (array): The input array with random blocks removed.

**highBlocking(input, blockLevel):**
Blocks a high portion of the input data by zeroing out an initial segment.

    Parameters:
        input (array): The input data.
        blockLevel (float): The proportion of the data to block from the beginning.

    Returns:
        blocked (array): The input array with a high portion blocked.

**preprocessing(img, dim=128):**
Preprocesses an image by resizing it and converting it to bipolar format.

    Parameters:
        img (array): The image to preprocess.
        dim (int, optional): The dimension to resize the image to (default is 128).

    Returns:
        flatty (array): The preprocessed and flattened image data.

**reshape(data):**
Reshapes the flattened data into a 2D square.

    Parameters:
        data (array): The flattened data.

    Returns:
        reshaped (array): The reshaped 2D array.

**comparePatterns(pat1, pat2):**
Compares two patterns and returns the proportion of matching elements.

    Parameters:
        pat1 (array): The first pattern.
        pat2 (array): The second pattern.

    Returns:
        similarity (float): The proportion of matching elements.

**getAccuracy(originals, finalised):**
Calculates the accuracy by comparing the original and final patterns.

    Parameters:
        originals (list of arrays): The list of original patterns.
        finalised (list of lists of arrays): The list of predicted patterns.

    Returns:
        accuracy (float): The accuracy based on correct predictions.

**resultsPlotter(original, iterations):**
Plots the original and predicted patterns in a grid format for visualization.

    Parameters:
        original (list of arrays): The list of original patterns.
        iterations (list of lists of arrays): The list of predicted patterns.

    Returns:
        None. Plots the results using matplotlib.pyplot.

###Main Process:

Load Images: Loads images from a specified directory and preprocesses them to be used with the Hopfield network.
Corrupt Images: Applies corruption (e.g., highBlocking) to the images to simulate damage.
Hopfield Network Prediction: Uses a Hopfield-based class (e.g., DAMDiscreteHopfield) to predict patterns from corrupted images.
Plot Results: Plots the original and predicted patterns to visually assess the results.

##testFramework.py

###Functions:

**randomFlipping(input, flipCount):**
Randomly flips elements in an input array based on a specified flip probability.

    Purpose: Simulate data corruption by randomly inverting values in an array.
    Parameters:
        input (array): The input array.
        flipCount (float): The probability of flipping an element.
    Returns: flippy (array): The modified array with random flips.

**highBlocking(input, blockLevel):**
Blocks a high proportion of an input array by setting a segment of elements to 1.

    Purpose: Simulate data corruption by blocking a portion of the input array.
    Parameters:
        input (array): The input array.
        blockLevel (float): The proportion of elements to block.
    Returns: blocked (array): The modified array with a high proportion blocked.

**GeneralErrorstuff(filename, HopfieldType, nums_neurons=[100], thetas=[0.0], corruption=[0,50,10], max_patterns=[1,50,1], betas=[8], rectified=True, powers=[2], pairwise_connections=False, max_error=1):**
Generalized test function to evaluate different Hopfield-type networks with varying parameters and corruption levels.

    Purpose: Test various Hopfield network configurations and generate error statistics.
    Parameters:
        filename (str): Name of the output file for test results.
        HopfieldType (str): Type of Hopfield network to use.
        nums_neurons (list of int): List of neuron counts to test.
        thetas (list of float): List of theta values for energy calculation.
        corruption (list of int): Corruption level details [min, max, step].
        max_patterns (list of int): Range for the number of patterns [min, max, step].
        betas (list of int): List of beta values for continuous Hopfield networks.
        rectified (bool): Whether to use rectification in energy calculation (DAM).
        powers (list of int): List of power values for DAM energy function.
        pairwise_connections (bool): If True, use pairwise connections for Simplicial Hopfield.
        max_error (float): Maximum error rate threshold to stop testing.
    Returns: None. Writes results to a CSV file and displays console output with test statistics.

**HopfieldSyncTests():**
Runs a series of synchronized Hopfield tests and outputs results to a CSV file.

    Purpose: Evaluate the performance of synchronized Hopfield networks with different neuron and pattern counts.
    Parameters: None.
    Returns: None. Writes test results to a CSV file and generates a scatter plot.

**HopfieldAsyncTests():**
Runs a series of asynchronous Hopfield tests and outputs results to a CSV file.

    Purpose: Evaluate the performance of asynchronous Hopfield networks with different neuron and pattern counts.
    Parameters: None.
    Returns: None. Writes test results to a CSV file and generates a scatter plot.

**DAMTests():**
Runs a series of Dense Associative Memory (DAM) tests and outputs results to a CSV file.

    Purpose: Evaluate the performance of DAM networks with different neuron and pattern counts.
    Parameters: None.
    Returns: None. Writes test results to a CSV file and generates a scatter plot.

###Script Execution:

The main section of the script allows for execution of various tests based on the configuration in GeneralErrorstuff. It also includes commented-out sections for previous experiments and outlines current or future test plans.

##plotResults.py

###Functions:

**plot_corruption(filename, title):**
Plots the error rate against the number of patterns for different corruption levels from a CSV file.

    Purpose: Visualize the error rate versus the number of patterns for various levels of data corruption.
    Parameters:
        filename (str): Path to the CSV file containing test results.
        title (str): Title for the plot.
    Returns: None. Displays a plot using matplotlib.pyplot.

**plot_thetas(filename, title, levels):**
Plots the error rate against the number of patterns for different theta values.

    Purpose: Visualize the error rate versus the number of patterns for different values of the parameter theta.
    Parameters:
        filename (str): Path to the CSV file containing test results.
        title (str): Title for the plot.
        levels (list of float): List of theta values to plot.
    Returns: None. Displays a plot using matplotlib.pyplot.

**plot_betas(filename, title, levels):**
Plots the error rate against the number of patterns for different beta values.

    Purpose: Visualize the error rate versus the number of patterns for different values of the parameter beta.
    Parameters:
        filename (str): Path to the CSV file containing test results.
        title (str): Title for the plot.
        levels (list of float): List of beta values to plot.
    Returns: None. Displays a plot using matplotlib.pyplot.

**plotAtCorruptionLevel(filename, title, corruption_level, param_vals):**
Plots the error rate against the number of patterns at a specific level of corruption for different parameter values.

    Purpose: Visualize the error rate versus the number of patterns at a specific level of corruption for different parameter values.
    Parameters:
        filename (str): Path to the CSV file containing test results.
        title (str): Title for the plot.
        corruption_level (float): The level of corruption to consider.
        param_vals (list of float): List of parameter values to plot.
    Returns: None. Displays a plot using matplotlib.pyplot.

plotAtCorruptionLevelMulti(filenames, title, corruption_level, param_vals, nums_neurons, labels, error_level):
Plots the error rate against the pattern-to-neuron ratio at a specific level of corruption for multiple files.

    Purpose: Visualize the error rate against the pattern-to-neuron ratio at a specific level of corruption for multiple configurations.
    Parameters:
        filenames (list of str): List of CSV files to process.
        title (str): Title for the plot.
        corruption_level (float): The level of corruption to consider.
        param_vals (list of lists of float): List of parameter values to plot.
        nums_neurons (list of int): List of neuron counts for each configuration.
        labels (list of str): Labels for each configuration.
        error_level (float): Error rate threshold for additional plot information.
    Returns: None. Displays a plot using matplotlib.pyplot.

**append_columns(infile, outfile, new_columns):**
Appends new columns to each line in a file.

    Purpose: Append additional information to a CSV file.
    Parameters:
        infile (str): Input file path.
        outfile (str): Output file path.
        new_columns (str): New column content to append to each line.
    Returns: None. Writes the modified content to the output file.

**plot_to_errorlever(filename, title, error_level, param_vals):**
Plots the critical number of patterns to maintain a specific error rate at different parameter values.

    Purpose: Identify the critical pattern count to maintain a specified error rate at different parameter values.
    Parameters:
        filename (str): Path to the CSV file containing test results.
        title (str): Title for the plot.
        error_level (float): The acceptable error rate threshold.
        param_vals (list of float): List of parameter values to plot.
    Returns: None. Displays a plot using matplotlib.pyplot.

**errorleverMulti(filenames, error_level, param_vals, corruption_level):**
Returns a list of critical pattern counts for multiple files, given an error rate threshold and a specific corruption level.

    Purpose: Calculate the critical pattern count to maintain a specific error rate at different parameter values for multiple configurations.
    Parameters:
        filenames (list of str): List of CSV files to process.
        error_level (float): The acceptable error rate threshold.
        param_vals (list of lists of float): List of parameter values to consider.
        corruption_level (float): The level of corruption to consider.
    Returns: critical_pattern_nums (list of float): The list of critical pattern counts for each configuration.

**writeTableToFile(outfile, filenames, Error_levels, param_vals, corruption_levels):**
Writes a table of critical pattern counts to a CSV file, based on error rate thresholds and corruption levels.

    Purpose: Generate a summary table of critical pattern counts for various error levels and corruption levels.
    Parameters:
        outfile (str): Output file path.
        filenames (list of str): List of CSV files to process.
        Error_levels (list of float): List of error rate thresholds.
        param_vals (list of lists of float): List of parameter values for each configuration.
        corruption_levels (list of float): List of corruption levels to consider.
    Returns: None. Writes the summary table to a CSV file.


