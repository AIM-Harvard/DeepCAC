require(ICC)
require(Daim)
require(pROC)

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
# Events_NLST.csv: This file needs to be prepared using data from NLST. It needs a PID column with
#   matching IDs to the results and an event column
NLST.results = read.csv(file='Results_NLST.csv')
NLST.events = read.csv(file='Events_NLST.csv')

# ICC
NLST.data = data.frame(CAC = c(NLST.results$CAC_AI, NLST.results$CAC_manual),
                       id = factor(NLST.results$PID))
ICCest(x = id, y = CAC, data = NLST.data)
cor.test(NLST.results$CAC_AI, NLST.results$CAC_manual, method="spearman")

# AUC
data.full = merge(x=NLST.results, y=NLST.events, by.x='PID', by.y='PID')
data.full$event.f = factor(data.full$event)
deLong.test(data.full[,c("CAC_AI","CAC_manual")], data.full$event, labpos="1")
roc(data.full$event.f, data.full$CAC_AI)
