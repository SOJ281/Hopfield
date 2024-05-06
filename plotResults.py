from matplotlib import pyplot as plt

file = open("Results\HopfieldErrorCorruption3.csv",'r')

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
plt.title("Error Rate vs. % Corruption of Original Image")
plt.xlabel("Number of Patterns")
plt.ylabel("Error Rate")
plt.legend()
plt.show()