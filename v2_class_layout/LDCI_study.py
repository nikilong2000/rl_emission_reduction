#1.Loading libraries
from LDCI_Layout import LDCI_Layout

#available in case we want to loop through cycle type
cycle_list=['Niederdynamisch.csv',\
            'NordRoute.csv', \
            'Renningen.csv', \
            'Stadtfocus.csv',\
            'Tuebingen0716.csv',\
            'Tuebingen0723.csv',\
            'Verbrauchsarm.csv']


LDCI_case=LDCI_Layout()

#set and load cycle
drivingcycle_folder="Driving_cycles_CSV"
drivingcycle_file="Tuebingen0716.csv" #could be taken iteratively from the list above
LDCI_case.set_cycle(drivingcycle_folder, drivingcycle_file)

#set control 
K_split=0.5
PCI=43250
rend=0.28

LDCI_case.set_basic_control(K_split,PCI,rend)

#set and load model M0, its scalers and first inputs
folder='./UStuttgart_model'
fileroot_model='model'
fileroot_scalerIn='input_scaler'
fileroot_scalerOut='output_scaler'

LDCI_case.set_model(folder,fileroot_model,fileroot_scalerIn,fileroot_scalerOut)


#execution along the cycle
LDCI_case.run()

#plotting
LDCI_case.plot()

#next steps: 
#1.adding statistical analysis to come up with some indicators
#2.add them into a table, to compare different cases
#3.export the table as a figure


