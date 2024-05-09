from matplotlib import pyplot as plt
import numpy as np
import time

def plot_corruption(filename, title):
    file = open(filename,'r')

    pattern_nums = []
    ErrorRates = []
    CorruptionLevels = []
    for line in file.readlines()[1:]:
        line = line.strip().split(',')
        pattern_nums.append(int(line[0]))
        ErrorRates.append(float(line[1]))
        CorruptionLevels.append(float(line[2]))

    for level in [0.0,0.1,0.2,0.3,0.4,0.5]:#set(CorruptionLevels):
        level_pattern_nums = []
        level_ErrorRates = []
        for i in range(len(pattern_nums)):
            if CorruptionLevels[i] == level:
                level_pattern_nums.append(pattern_nums[i])
                level_ErrorRates.append(ErrorRates[i])
        if level == 0.0:
            plt.plot(level_pattern_nums, level_ErrorRates, label="%s%% corruption" % (float(level)*100), linestyle='dashed')
        else:
            plt.plot(level_pattern_nums, level_ErrorRates, label="%s%% corruption" % (float(level)*100))
    plt.title(title)
    plt.xlabel("Number of Patterns")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.show()

def plot_thetas(filename, title, levels):
    file = open(filename,'r')

    pattern_nums = []
    ErrorRates = []
    CorruptionLevels = []
    thetas = []
    for line in file.readlines()[1:]:
        line = line.strip().split(',')
        pattern_nums.append(int(line[0]))
        ErrorRates.append(float(line[1]))
        CorruptionLevels.append(float(line[2]))
        thetas.append(float(line[4]))

    for level in levels:#set(CorruptionLevels):
        level_pattern_nums = []
        level_ErrorRates = []
        for i in range(len(pattern_nums)):
            if thetas[i] == level:
                level_pattern_nums.append(pattern_nums[i])
                level_ErrorRates.append(ErrorRates[i])
        if level == 0.0:
            plt.plot(level_pattern_nums, level_ErrorRates, label="theta = %s" % (float(level)), linestyle='dashed')
        else:
            plt.plot(level_pattern_nums, level_ErrorRates, label="theta = %s" % (float(level)))
    plt.title(title)
    plt.xlabel("Number of Patterns")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.show()

def plot_betas(filename, title, levels):
    file = open(filename,'r')

    pattern_nums = []
    ErrorRates = []
    CorruptionLevels = []
    thetas = []
    for line in file.readlines()[1:]:
        line = line.strip().split(',')
        pattern_nums.append(int(line[0]))
        ErrorRates.append(float(line[1]))
        CorruptionLevels.append(float(line[2]))
        thetas.append(float(line[4]))

    for level in levels:#set(CorruptionLevels):
        level_pattern_nums = []
        level_ErrorRates = []
        for i in range(len(pattern_nums)):
            if thetas[i] == level:
                level_pattern_nums.append(pattern_nums[i])
                level_ErrorRates.append(ErrorRates[i])
        if level == 0.0:
            plt.plot(level_pattern_nums, level_ErrorRates, label="beta = %s" % (float(level)), linestyle='dashed')
        else:
            plt.plot(level_pattern_nums, level_ErrorRates, label="beta = %s" % (float(level)))
    plt.title(title)
    plt.xlabel("Number of Patterns")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.show()

def plotAtCorruptionLevel(filename, title, corruption_level, param_vals):
    file = open(filename,'r')

    pattern_nums = []
    ErrorRates = []
    CorruptionLevels = []
    param = []
    for line in file.readlines()[1:]:
        line = line.strip().split(',')
        if float(line[2]) == corruption_level:
            pattern_nums.append(int(line[0]))
            ErrorRates.append(float(line[1]))
            param.append(float(line[4])) # The parameter being changed e.g. theta, beta

    for index,val in enumerate(param_vals):
        param_pattern_nums = []
        param_ErrorRates = []
        for p in range(len(pattern_nums)):
            if param[p] == val:
                param_pattern_nums.append(pattern_nums[p])
                param_ErrorRates.append(ErrorRates[p])

        print(val)

        if param == 0.0:
            plt.plot(param_pattern_nums, param_ErrorRates, label="%s" % val, linestyle='dashed')
        else:
            plt.plot(param_pattern_nums, param_ErrorRates, label="%s" % val)
    
    plt.title(title)
    plt.xlabel("Number of Patterns")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.show()

def plotAtCorruptionLevelMulti(filenames, title, corruption_level, param_vals, nums_neurons, labels, error_level):
    for file_index, filename in enumerate(filenames):
        file = open(filename,'r')

        pattern_ratios = []
        ErrorRates = []
        CorruptionLevels = []
        param = []
        for line in file.readlines()[1:]:
            line = line.strip().split(',')
            if float(line[2]) == corruption_level:
                pattern_ratios.append(float(line[0])/float(line[3]))
                ErrorRates.append(float(line[1]))
                param.append(float(line[4])) # The parameter being changed e.g. theta, beta

        for index,val in enumerate(param_vals[file_index]):
            param_pattern_nums = []
            param_ErrorRates = []
            for p in range(len(pattern_ratios)):
                if param[p] == val:
                    param_pattern_nums.append(pattern_ratios[p])
                    param_ErrorRates.append(ErrorRates[p])

            plt.plot(param_pattern_nums, param_ErrorRates, label="%s" % labels[file_index])
    
    plt.axhline(y=error_level, color='r', linestyle='dashed')
    plt.title(title)
    plt.xlabel("Ratio of Patterns:Neurons")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.show()

def append_columns(infile, outfile,new_collumns):
    inf = open(infile,'r')
    outf = open(outfile,'w')

    for line in inf.readlines():
        new_line=line.strip()+new_collumns
        outf.write(new_line+"\n")

    inf.close()
    outf.close()

def plot_to_errorlever(filename,title,error_level,param_vals):
    file = open(filename,'r')
    pattern_nums = []
    ErrorRates = []
    CorruptionLevels = []
    param = []
    for line in file.readlines()[1:]:
        line = line.strip().split(',')
        pattern_nums.append(int(line[0]))
        ErrorRates.append(float(line[1]))
        CorruptionLevels.append(float(line[2]))
        param.append(float(line[4])) # The parameter being changed e.g. theta, beta

    # Plot a line for the critical number of patterns to maintain acceptable error rate at each corruption level
    for corruption_level in [0.0,0.1,0.2,0.3,0.4,0.5]:
        critical_pattern_nums = []
        for val in param_vals:
            reached_threshold = False
            val_pattern_nums = []
            val_ErrorRates = []
            for i in range(len(pattern_nums)):
                if param[i] == val and CorruptionLevels[i] == corruption_level:
                    val_pattern_nums.append(pattern_nums[i])
                    val_ErrorRates.append(ErrorRates[i])
            
            for j in range(1,len(val_pattern_nums)):
                print(val_ErrorRates[j], error_level, val_ErrorRates[j] >= error_level)
                if val_ErrorRates[j] >= error_level:
                    # plot this as the critical point for the network
                    critical_pattern_nums.append(val_pattern_nums[j]) # j-1 to get the pattern BEFORE it crosses the error rate crosses the threshold
                    reached_threshold = True
                    break

            if reached_threshold == False:
                critical_pattern_nums.append(np.nan)

        print(param_vals)
        print(critical_pattern_nums)
        plt.plot(param_vals, critical_pattern_nums, label="%s%% corruption" % (float(corruption_level)*100), marker='o')

    plt.title(title)
    plt.xlabel("Param value")
    plt.ylabel("Number of Patterns stored before Error Rate exceeds %s" % error_level)
    plt.legend()
    plt.show()

def errorleverMulti(filenames,error_level,param_vals,Corruption_level):
    critical_pattern_nums = []
    for file_index, filename in enumerate(filenames):
        file = open(filename,'r')
        pattern_nums = []
        ErrorRates = []
        CorruptionLevels = []
        param = []
        for line in file.readlines()[1:]:
            line = line.strip().split(',')
            pattern_nums.append(float(line[0])/float(line[3]))
            ErrorRates.append(float(line[1]))
            CorruptionLevels.append(float(line[2]))
            param.append(float(line[4])) # The parameter being changed e.g. theta, beta

        # Plot a line for the critical number of patterns to maintain acceptable error rate at each corruption level
        for corruption_level in [Corruption_level]:
            for val in param_vals[file_index]:
                reached_threshold = False
                val_pattern_nums = []
                val_ErrorRates = []
                for i in range(len(pattern_nums)):
                    #print(param[i],val,CorruptionLevels[i],corruption_level)
                    if param[i] == val and CorruptionLevels[i] == corruption_level:
                        val_pattern_nums.append(pattern_nums[i])
                        val_ErrorRates.append(ErrorRates[i])
                
                for j in range(1,len(val_pattern_nums)):
                    #print(val_ErrorRates[j], error_level, val_ErrorRates[j] >= error_level)
                    if val_ErrorRates[j] >= error_level:
                        # plot this as the critical point for the network
                        critical_pattern_nums.append(val_pattern_nums[j-1]) # j-1 to get the pattern BEFORE it crosses the error rate crosses the threshold
                        reached_threshold = True
                        break

                if reached_threshold == False:
                    critical_pattern_nums.append(np.nan)

            # plt.plot(param_vals, critical_pattern_nums, label="%s%% corruption" % (float(corruption_level)*100), marker='o')

    # plt.title(title)
    # plt.xlabel("Param value")
    # plt.ylabel("Number of Patterns stored before Error Rate exceeds %s" % error_level)
    # plt.legend()
    # plt.show()   
    return critical_pattern_nums

def writeTableToFile(outfile,filenames,Error_levels,param_vals,corruption_levels):
    t = time.localtime(time.time())
    formatted_t = (''.join((str(t.tm_mday),str(t.tm_hour), str(t.tm_min))))
    file=open(outfile+formatted_t+".csv",'x')
    
    file.write("Corruption_Level,Error_Level,"+','.join(labels)+"\n")
    for corruption_level in corruption_levels:
        for Error_level in Error_levels:
            critical_nums = [str(val) for val in errorleverMulti(filenames,Error_level,param_vals,corruption_level)]
            file.write(str(corruption_level)+","+str(Error_level)+","+','.join(critical_nums)+"\n")
    file.close()

if __name__ == '__main__':
    #append_columns("Results\HopfieldError.csv","Results\HopfieldError_withthetas.csv","100,0.0")
    
    #plot_thetas("Results\HopfieldErrorNoCorruptionThetas62136.csv", "Error Rate vs. value of theta (no corruption)", [0.0,0.1,0.2,0.4,0.8])
    #plot_corruption("Results\HopfieldErrorCorruption3.csv", "Hopfield: Error Rate vs. % Corruption of Original Image")
    #plot_corruption("Results\HopfieldErrorDAM7139.csv", "DAM: Error Rate vs. % Corruption of Original Image")
    # plot_corruption("Results\HopfieldErrorContinuous71014.csv", "Continuous: Error Rate vs. % Corruption of Original Image")
    # plot_betas("Results\HopfieldErrorContinuousDifferentbetas71335.csv", "Error Rate vs. value of beta (no corruption)", [0.0,0.25,0.5,1.0,2.0,4.0])

    #plot_corruption("Results\HopfieldErrorContinuousBinary71912.csv", "Continuous: Error Rate vs. % Corruption of Original Image")
    #plot_betas("Results\HopfieldErrorContinuousDifferentbetas72020.csv", "Error Rate vs. value of beta (no corruption)", [0.0,0.25,0.5,1.0,2.0,4.0])
    
    #plot_thetas("Results\HopfieldErrorNoCorruptionThetas62136.csv", "Error Rate vs. value of theta (no corruption)")
    #plot_corruption("Results\HopfieldErrorCorruption3.csv", "Hopfield: Error Rate vs. % Corruption of Original Image")
    #plot_corruption("Results\HopfieldErrorDAM7139.csv", "DAM: Error Rate vs. % Corruption of Original Image")
    #plot_corruption("Results\HopfieldErrorContinuousDifferentNinjas72325.csv", "Continuous: Error Rate vs. % Corruption of Original Image")


    """
    """
    # New Results
    """
    # Different betas, 0-50% corruption
    #plot_betas("Results\Scott\HopfieldErrorContinuous8103.csv", "Error Rate vs. value of beta", [0.0,0.25,0.5,1.0,2.0,4.0])
    #plot_betas("Results\Scott\HopfieldErrorContinuousBinary8517.csv", "Error Rate vs. value of beta", [0.0,0.25,0.5,1.0,2.0,4.0])

    #   Predicting Continuous Values
    plotAtCorruptionLevel("Results\Scott\HopfieldErrorContinuous8103.csv", "Continuous: Error rate vs. value of beta (no corruption)", corruption_level=0.0, param_vals=[1,2,4,8,16,32,64])
    plotAtCorruptionLevel("Results\Scott\HopfieldErrorContinuous8103.csv", "Continuous: Error rate vs. value of beta (20% corruption)", corruption_level=0.2, param_vals=[1,2,4,8,16,32,64])
    plotAtCorruptionLevel("Results\Scott\HopfieldErrorContinuous8103.csv", "Continuous: Error rate vs. value of beta (50% corruption)", corruption_level=0.5, param_vals=[1,2,4,8,16,32,64])

    # Predicting Binary Values
    plotAtCorruptionLevel("Results\Scott\HopfieldErrorContinuousBinary8517.csv", "ContinuousBinary: Error rate vs. value of beta (no corruption)", corruption_level=0.0, param_vals=[64,128,256])
    plotAtCorruptionLevel("Results\Scott\HopfieldErrorContinuousBinary8517.csv", "ContinuousBinary: Error rate vs. value of beta (20% corruption)", corruption_level=0.2, param_vals=[64,128,256])
    plotAtCorruptionLevel("Results\Scott\HopfieldErrorContinuousBinary8517.csv", "ContinuousBinary: Error rate vs. value of beta (50% corruption)", corruption_level=0.5, param_vals=[64,128,256])

    #plot_corruption("Results\Scott\HopfieldErrorContinuous8103.csv", "Error rate vs. value of beta (no corruption)")


    # Different powers, 0-50% corruption
    #plot_betas("Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial8521.csv", "Error Rate vs. value of power", [0.0,0.25,0.5,1.0,2.0,4.0])
    #plot_betas("Results\Scott\HopfieldErrorDAMEXPFull81049.csv", "Error Rate vs. value of power", [0.0,0.25,0.5,1.0,2.0,4.0])
    
    #   Polynomial
    plotAtCorruptionLevel("Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial8521.csv", "DAMPolynomial: Error rate vs. value of beta (no corruption)", corruption_level=0.0, param_vals=[1,2,4,8,16,32,64])
    plotAtCorruptionLevel("Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial8521.csv", "DAMPolynomial: Error rate vs. value of beta (20% corruption)", corruption_level=0.2, param_vals=[1,2,4,8,16,32,64])
    plotAtCorruptionLevel("Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial8521.csv", "DAMPolynomial: Error rate vs. value of beta (50% corruption)", corruption_level=0.5, param_vals=[1,2,4,8,16,32,64])
    #   Rectified Polynomial
    plotAtCorruptionLevel("Results\HopfieldErrorDAMDifferentPowerRectified8243.csv", "DAMRectified: Error rate vs. value of beta (no corruption)", corruption_level=0.0, param_vals=[1,2,4,8,16,32,64])
    plotAtCorruptionLevel("Results\HopfieldErrorDAMDifferentPowerRectified8243.csv", "DAMRectified: Error rate vs. value of beta (20% corruption)", corruption_level=0.2, param_vals=[1,2,4,8,16,32,64])
    plotAtCorruptionLevel("Results\HopfieldErrorDAMDifferentPowerRectified8243.csv", "DAMRectified: Error rate vs. value of beta (50% corruption)", corruption_level=0.5, param_vals=[1,2,4,8,16,32,64])
    #   Exponential
    plot_corruption("Results\Scott\HopfieldErrorDAMEXPFull81049.csv", "DAMExp: Error rate vs. value of beta (no corruption)")
    plot_corruption("Results\Scott\HopfieldErrorDAMEXPFull81049.csv", "DAMExp: Error rate vs. value of beta (20% corruption)")
    plot_corruption("Results\Scott\HopfieldErrorDAMEXPFull81049.csv", "DAMExp: Error rate vs. value of beta (50% corruption)")
    """

    #plot_to_errorlever("Results\Scott\HopfieldErrorContinuous8103.csv","Critical numbers of patterns to maintain Error Rate <= %s" % error_level,error_level=error_level,param_vals=[1,2,4,8,16,32,64])


    #append_columns("Results\HopfieldErrorCorruption3.csv","Results\HopfieldErrorCorruption3withparams.csv",",100,0.0")
    # Error rate, for a fixed hyperparameter value, at a fixed level of corruption, for each Hopfield type on the same graph
    # If you need more results could run more experiments set to cut out it ErrorRate goes above threshold (or a certain time has elapsed)
    #plotAtCorruptionLevel("Results\HopfieldErrorDAMDifferentPowerRectified8243.csv", "DAMRectified: Error rate vs. value of beta (10% corruption)", corruption_level=0.1, param_vals=[64])
    #filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\HopfieldErrorDAMDifferentPowerRectified8243.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial8521.csv", "Results\Scott\HopfieldErrorDAMEXPFull81049.csv","Results\Scott\HopfieldErrorContinuousBinary8517.csv","Results\Scott\HopfieldErrorContinuous8103.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv"]
    #filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial81910.csv", "Results\Scott\HopfieldErrorDAMEXPFull81049.csv","Results\Scott\HopfieldErrorContinuousBinary8517.csv","Results\Scott\HopfieldErrorContinuous8103.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    #filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv","Results\Scott\HopfieldErrorDAMEXPFull81049.csv","Results\Scott\HopfieldErrorContinuousBinary8517.csv","Results\Scott\HopfieldErrorContinuous8103.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    
    error_level = 0.1
    corruption_levels = [0.0,0.1]
    filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial81910.csv", "Results\Scott\HopfieldErrorDAMEXPFull81049.csv","Results\Scott\HopfieldErrorContinuousBinary8517.csv","Results\Scott\HopfieldErrorContinuous8103.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    #filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial81910.csv", "Results\Scott\HopfieldErrorDAMExponential1st9448.csv","Results\HopfieldErrorContinuousBinaryInversion9315.csv","Results\HopfieldErrorContinuousInversion9350.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial81910.csv", "Results\Scott\HopfieldErrorDAMEXPFull81049.csv","Results\HopfieldErrorContinuousBinaryInversion29517.csv","Results\HopfieldErrorContinuousInversion2952.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    nums_neurons = [100,100,100,100,100,100,100]
    param_vals = [[0],[2],[2],[16],[64],[64],[0],[0]]
    labels = ["Hopfield", "DAM", "DAMPoly","DAMExp", "ContinuousBinary", "Continuous", "SimplicialWithPairwise", "SimplicialNoPairwise"]
    #regular(theta=0), DAMrectified(power), DAMpoly(power), DAMExp(), ContinuousBinary(beta), Continuous(beta), Simplicial 
    # plotAtCorruptionLevelMulti(filenames, "Error Rate by Hopfield Type (%s%% corruption)" % (corruption_levels[0]*100), corruption_level=corruption_levels[0], param_vals=param_vals,nums_neurons=nums_neurons, labels=labels,error_level=error_level)
    #plotAtCorruptionLevelMulti(filenames, "Error Rate by Hopfield Type (%s%% corruption)" % (corruption_levels[1]*100), corruption_level=corruption_levels[1], param_vals=param_vals,nums_neurons=nums_neurons, labels=labels,error_level=error_level)

    # filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial81910.csv", "Results\Scott\HopfieldErrorDAMExponential1st9448.csv","Results\HopfieldErrorContinuousBinaryInversion9315.csv","Results\HopfieldErrorContinuousInversion9350.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial81910.csv", "Results\Scott\HopfieldErrorDAMExponential1st9448.csv","Results\HopfieldErrorContinuousBinaryInversion29517.csv","Results\HopfieldErrorContinuousInversion2952.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    error_level = [0.0,0.025,0.05,0.1]
    corruption_levels = [0.0,0.1]
    #writeTableToFile("CriticalValuesTable",filenames,error_level,param_vals,corruption_levels)


    # Reduced number of network types
    error_level = 0.1
    corruption_levels = [0.0,0.1]
    filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial81910.csv", "Results\Scott\HopfieldErrorDAMEXPFull81049.csv","Results\Scott\HopfieldErrorContinuousBinary8517.csv","Results\Scott\HopfieldErrorContinuous8103.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    #filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial81910.csv", "Results\Scott\HopfieldErrorDAMExponential1st9448.csv","Results\HopfieldErrorContinuousBinaryInversion9315.csv","Results\HopfieldErrorContinuousInversion9350.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMEXPFull81049.csv","Results\HopfieldErrorContinuousInversion2952.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    nums_neurons = [100,100,100,100,100,100,100]
    param_vals = [[0],[2],[16],[64],[0],[0]]
    labels = ["Hopfield", "DAM","DAMExp", "Continuous", "PairwiseSimplicial", "Simplicial"]
    #regular(theta=0), DAMrectified(power), DAMpoly(power), DAMExp(), ContinuousBinary(beta), Continuous(beta), Simplicial 
    plotAtCorruptionLevelMulti(filenames, "Error Rate by Hopfield Type (%s%% corruption)" % (corruption_levels[0]*100), corruption_level=corruption_levels[0], param_vals=param_vals,nums_neurons=nums_neurons, labels=labels,error_level=0.025)
    #plotAtCorruptionLevelMulti(filenames, "Error Rate by Hopfield Type (%s%% corruption)" % (corruption_levels[1]*100), corruption_level=corruption_levels[1], param_vals=param_vals,nums_neurons=nums_neurons, labels=labels,error_level=error_level)

    # filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerPolynomial81910.csv", "Results\Scott\HopfieldErrorDAMExponential1st9448.csv","Results\HopfieldErrorContinuousBinaryInversion9315.csv","Results\HopfieldErrorContinuousInversion9350.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    filenames = ["Results\HopfieldErrorCorruption3withparams.csv", "Results\Scott\HopfieldErrorDAMDifferentPowerRectified81634with2.csv","Results\Scott\HopfieldErrorDAMExponential1st91051.csv","Results\HopfieldErrorContinuousInversion2952.csv", "Results\HopfieldErrorSimplicialPairwise81937.csv", "Results\HopfieldErrorSimplicialNopairwise81933.csv"]
    param_vals = [[0],[2],[0],[64],[0],[0]]
    error_level = [0.0,0.025,0.05,0.1,0.25]
    corruption_levels = [0.0,0.1]
    #writeTableToFile("CriticalValuesTable",filenames,error_level,param_vals,corruption_levels)