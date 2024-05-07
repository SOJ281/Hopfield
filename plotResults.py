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

def plot_thetas(filename, title):
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

    for level in [0.0,0.1,0.2,0.4,0.8]:#set(CorruptionLevels):
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
    #plot_thetas("Results\HopfieldErrorNoCorruptionThetas62136.csv", "Error Rate vs. value of theta (no corruption)")
    #plot_corruption("Results\HopfieldErrorCorruption3.csv", "Hopfield: Error Rate vs. % Corruption of Original Image")
    #plot_corruption("Results\HopfieldErrorDAM7139.csv", "DAM: Error Rate vs. % Corruption of Original Image")
    plot_corruption("Results\HopfieldErrorContinuousDifferentNinjas72325.csv", "Continuous: Error Rate vs. % Corruption of Original Image")