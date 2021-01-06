library(survival)

# The code and data of this repo are intended to promote reproducible research of the paper
# "Deep convolutional neural networks to predict cardiovascular risk from computed tomography"
# Details about the project can be found at our project webpage at 
# https://aim.hms.harvard.edu/deepcac

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Two files are needed:
# Results_NLST.csv: This is provided
# Data_NLST.csv: This file needs to be prepared using data from NLST. It needs a PID column with
#   matching IDs to the results and the following columns provided by NLST: fup_days, deathstat,
#   weight, height, dcficd

raw.data.results = read.csv(file = 'Results_NLST.csv')
raw.data.nlst = read.csv(file = 'Data_NLST.csv')
data.full = merge(x=raw.data.results, y=raw.data.nlst, by.x='PID', by.y='PID')

# Calculate death days
data.full$survivaldays = data.full$fup_days
data.full$survivaldays[which(data.full$deathstat == 0)] = 
  data.full$survivaldays[which(data.full$deathstat == 0)] - 58.1772
data.full$survivaldays[which(data.full$survivaldays < 0)] = 0
data.full$survivaldays[which(is.na(data.full$survivaldays))] = 7*365

# Calculate BMI and obese
data.full$BMI = NA
data.full$BMI = data.full$weight / (data.full$height * data.full$height) * 703
data.full$obese = NA
data.full$obese[which(data.full$BMI>30)] = 1
data.full$obese[which(data.full$BMI<=30)] = 0
data.full$obese = factor(data.full$obese)

# Calculate CAC classes
data.full$classPred = NA
data.full$classPred = data.full$CAC_AI
data.full$classPred[which(data.full$classPred==0)] = 0
data.full$classPred[which(data.full$classPred>0 & data.full$classPred<=100)] = 1
data.full$classPred[which(data.full$classPred>100 & data.full$classPred<=300)] = 2
data.full$classPred[which(data.full$classPred>300)] = 3
data.full$classPred = factor(data.full$classPred)

# Get ASCVD events
data.full$ASCVD = 1
data.full$ASCVD[grep(pattern = "^I",x = as.character(data.full$dcficd))] = 2

# Cox un-adjusted ##################################################################################
res.cox <- coxph(Surv(survivaldays, ASCVD) ~ classPred, data = data.full)
res.summary = summary(res.cox)
cox.result = data.frame(round(res.summary$conf.int[,-2],2),
                        "p-value" = res.summary$coefficients[,"Pr(>|z|)"])
cox.result$CI = paste(round(cox.result$lower..95,2),"-",round(cox.result$upper..95,2),sep = "")
cox.result[,c("exp.coef.","CI","p.value")]

# Cox adjusted for age and sex #####################################################################
res.cox <- coxph(Surv(survivaldays, ASCVD) ~ classPred+age+gender, data = data.full)
res.summary = summary(res.cox)
cox.result = data.frame(round(res.summary$conf.int[,-2],2),
                        "p-value" = res.summary$coefficients[,"Pr(>|z|)"])
cox.result$CI = paste(round(cox.result$lower..95,2),"-",round(cox.result$upper..95,2),sep = "")
cox.result[,c("exp.coef.","CI","p.value")]

# Cox adjusted for everything ######################################################################
res.cox <- coxph(Surv(survivaldays, ASCVD) ~ 
                   classPred+age+gender+cigsmok_mbs+diagdiab+diaghype+obese, data = data.full)
res.summary = summary(res.cox)
cox.result = data.frame(round(res.summary$conf.int[,-2],2),
           "p-value" = res.summary$coefficients[,"Pr(>|z|)"])
cox.result$CI = paste(round(cox.result$lower..95,2),"-",round(cox.result$upper..95,2),sep = "")
cox.result[,c("exp.coef.","CI","p.value")]
