from matplotlib import pyplot as plt

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

def append_columns(infile, outfile,new_collumns):
    inf = open(infile,'r')
    outf = open(outfile,'w')

    for line in inf.readlines():
        new_line=line.strip()+new_collumns
        outf.write(new_line+"\n")

    inf.close()
    outf.close()

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
    New Results
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
