# Load packages
#install.packages("MatchIt")
#install.packages("readr")
library(MatchIt)
library(readr)

# Load data
DF_noVB <- read_csv("C:/Users/marij/OneDrive/Documenten/TM Jaar 2/TM Stage IDA/df_0_baseline.csv")
DF_VB <- read_csv("C:/Users/marij/OneDrive/Documenten/TM Jaar 2/TM Stage IDA/df_1_baseline.csv")
DF <- rbind(DF_noVB, DF_VB) # Combine the datasets

# Make gender factor
DF$Gender <- factor(DF$Gender)

# Matching
set.seed(123) # To get same sample if you run it multiple times
m <- matchit(ID_label ~ Age + Gender, method = "nearest", replace = F, ratio = 1, data = DF) # matched on age and gender
DF1 <- match.data(m)

table(DF1$ID_label)
tapply(DF1$Gender, DF1$ID_label, table)
tapply(DF1$Age, DF1$ID_label, summary)

chisq.test(DF1$Gender, DF1$ID_label)
t.test(DF1$Age ~ DF1$ID_label)

write_csv(DF1, "C:/Users/marij/OneDrive/Documenten/TM Jaar 2/TM Stage IDA/df_matched.csv")
