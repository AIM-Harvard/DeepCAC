library(survival)
library(survminer)

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
#   dcficd

raw.data.results = read.csv(file = 'Results_NLST.csv')
raw.data.nlst = read.csv(file = 'Data_NLST.csv')
data.full = merge(x=raw.data.results, y=raw.data.nlst, by.x='PID', by.y='PID')

# Calculate the death days
data.full$survivaldays = data.full$fup_days
data.full$survivaldays[which(data.full$deathstat == 0)] = 
  data.full$survivaldays[which(data.full$deathstat == 0)] - 58.1772
data.full$survivaldays[which(data.full$survivaldays < 0)] = 0
data.full$survivaldays[which(is.na(data.full$survivaldays))] = 7*365

# Recalculate cac classes
data.full$classPred = NA
data.full$classPred = data.full$CAC_AI
data.full$classPred[which(data.full$classPred==0)] = 0
data.full$classPred[which(data.full$classPred>0 & data.full$classPred<=100)] = 1
data.full$classPred[which(data.full$classPred>100 & data.full$classPred<=300)] = 2
data.full$classPred[which(data.full$classPred>300)] = 3
data.full$classPred = factor(data.full$classPred)

# Get ASCVD deaths
data.full$ASCVD = 1
data.full$ASCVD[grep(pattern = "^I",x = as.character(data.full$dcficd))] = 2

# KM CAC
fit <- survfit(Surv(survivaldays, ASCVD) ~ classPred, data = data.full)
res <- ggsurvplot(fit, data = data.full, risk.table = TRUE, conf.int = FALSE, censor = FALSE,
                  ylim = c(0.9,1), xlim=c(0,2310), xscale="d_y", break.x.by=365,
                  palette = c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"),
                  legend.labs = c("Very Low", "Low", "Moderate", "High"),
                  legend.title = "", legend = c(0.2, 0.2),
                  tables.theme = theme_cleantable(),
                  risk.table.height = 0.20, xlab = 'Time (Years)', ylab = 'ASCVD death free survival')
res$plot <- res$plot + theme(legend.key.width = unit(8, "mm"),
                             legend.key.height = unit(5, "mm"),
                             legend.text = element_text(size = 12))
print(res)
