#1.Loading libraries
import numpy as np #Mathematics: arrays
import matplotlib.pyplot as plt #Matplotlib

from PredictiveController import PredictiveController
from ExtendedEngine import ExtendedEngine

class LDCI_Layout:
    def __init__(self):
        self.MyEngine=ExtendedEngine()
        self.MyControl=PredictiveController()

        self.dataX, self.dataY, self.data_ctrl = [], [], [] #plot structures

        return

    def set_cycle(self,folder,file_cycle):
        self.MyControl.load_cycle(folder, file_cycle)
        return

    def set_basic_control(self,K_split,PCI,rend):
        self.MyControl.set_gains(K_split,PCI,rend)
        return


    def set_model(self,folder,fileroot_model,fileroot_scalerIn,fileroot_scalerOut):

        #at this stage I consider these constants as local, later could be moved to main file
        Nobserv=1
        Linput=1
        Nvar=6

        aux_par1=1
        aux_par2=1
        aux_par3=4

        self.MyEngine.load_model(folder, fileroot_model)
        self.MyEngine.load_scalers(folder, fileroot_scalerIn, fileroot_scalerOut)
        self.MyEngine.create_main_in(Nobserv,Linput,Nvar)
        self.MyEngine.create_aux_ini(aux_par1,aux_par2,aux_par3)

        return

    def run(self):

        aux_ini=self.MyEngine.get_aux_ini() #already covers the c==1 condition
        Ncase=self.MyControl.get_Ncase()
        vtime=self.MyControl.get_vect_time_cycle()
        #Ncase=50 #for checks/debug

        for c in range(1, Ncase):
            print('row %d: time=%.2f' % (c,vtime[c]))
            #control
            v_k,n_ice_sp,P_EM1,P_EM2,GP,m_inj=self.MyControl.control_loop(c)

            #engine
            self.MyEngine.set_main_in(v_k,n_ice_sp,P_EM1,P_EM2,GP,m_inj)
            v,vi,vo = self.MyEngine.run()

            #plot_data+others
            self.dataX.append(vi)
            self.dataY.append(vo)
            self.data_ctrl.append(self.MyControl.get_plot_data())
            self.MyEngine.set_aux_ini(v)

        return

    def plot_dataset(self,column_names, dataset):
        columns = {}
        for i, name in enumerate(column_names):
            columns[name] = dataset[:, i]
        # Creating subplots
        num_columns = len(columns)
        fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(num_columns,10))

        # Plotting each column in a separate subplot
        for i in range(num_columns):
            axes[i].plot(columns[column_names[i]])
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Value')
            axes[i].set_title(column_names[i])

        plt.tight_layout()
        #plt.show() #I prefer to see the three figures together, moved to plot()

        return

    def plot(self):

        #first plot: output variables
    
        n_rows = np.asarray(self.dataY).shape[0]
        n_cols = np.asarray(self.dataY).shape[2]
        Y = np.reshape(self.dataY, (n_rows, n_cols))

        column_names = ['SOC', 'NO', 'NO2', 'Torque']
        self.plot_dataset(column_names, Y)

        #second plot: input variables
        n_rows2 = np.asarray(self.dataX).shape[0]
        n_cols2 = np.asarray(self.dataX).shape[2]
        X = np.reshape(self.dataX, (n_rows2, n_cols2))

        column_names = ['v_k [km/h]', 'n_ice_sp [rpm]', 'P_EM1 [kW]', 'P_EM2 [kW]', 'Grade Profile [%]', 'm_inj [gps]']
        self.plot_dataset(column_names, X)

        
        #third plot: control variables
        n_rows3 = np.asarray(self.data_ctrl).shape[0]
        n_cols3 = np.asarray(self.data_ctrl).shape[1]
        C = np.reshape(self.data_ctrl, (n_rows3, n_cols3))

        column_names = ['P_req [kW]','ac_ped [%]','P_ice [kW]']

        self.plot_dataset(column_names, C)
        
        plt.show()

        #to be improved: position of figures to see all them at once

        return
        
