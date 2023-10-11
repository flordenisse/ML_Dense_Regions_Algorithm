import sys
from interface import Ui_MainWindow

from cProfile import label

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QAction, QApplication, QFileDialog, QMainWindow, QHBoxLayout
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtWidgets import QMessageBox
from PyQt5.sip import delete

from scipy.fftpack import ss_diff
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import SimpleITK as sitk
from SimpleITK import ConnectedThreshold, GetImageFromArray, GetArrayFromImage
import cv2
from matplotlib.figure import Figure

#Librerias para la densidad
from scipy import stats
from skimage import feature as ft
from skimage.feature import graycomatrix, graycoprops

#Librerias para métricas de texturas
from skimage.measure.entropy import shannon_entropy

#Librerias para entropia
from skimage.morphology import disk
from skimage.filters.rank import entropy

import scipy

#Librerias para ML
from sklearn.ensemble import RandomForestClassifier
import joblib



class Aplicacion(QMainWindow):

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()    
        self.ui.setupUi(self)

        #graficos

        self.grafico_a = plt.figure(1,frameon=False) #False
        self.a1 = self.grafico_a.add_subplot(111)
        self.grafico_b = plt.figure(2,frameon=True)
        self.b1 = self.grafico_b.add_subplot(111)
        self.densidad = plt.figure()
        self.entropia_1= plt.figure()
        self.entropia_4= plt.figure()
        self.densidad_3= plt.figure()

        self.a3 = self.entropia_1.add_subplot(111)
        self.a2 = self.densidad.add_subplot(111)
        self.a4 = self.entropia_4.add_subplot(111)
        self.a5 = self.densidad_3.add_subplot(111)


        self.grafico_a.patch.set_facecolor("#FFFFFF") #fondo de los gráficos: gris #353535 #FFFFFF
        self.grafico_b.patch.set_facecolor("#000000") #fondo de los gráfico
        self.densidad.patch.set_facecolor("#000000") 
        self.entropia_1.patch.set_facecolor("#000000") 
        self.entropia_4.patch.set_facecolor("#000000") 
        self.densidad_3.patch.set_facecolor("#000000") 
        
        
        self.canvas = FigCanvas(self.grafico_a) #monto la figura en el objeto canvas
        self.canvas2 = FigCanvas(self.grafico_b) #monto la figura en el objeto canvas
        self.canvas3 = FigCanvas(self.densidad) #monto la figura en el objeto canvas
        self.canvas4 = FigCanvas(self.entropia_1) #monto la figura en el objeto canvas
        self.canvas5 = FigCanvas(self.entropia_4)
        self.canvas6 = FigCanvas(self.densidad_3)

        self.ui.grafico_arriba.addWidget(self.canvas) 
        self.ui.grafico_abajo.addWidget(self.canvas2) 
        self.ui.densidad.addWidget(self.canvas3) 
        self.ui.entropia_1.addWidget(self.canvas4)
        self.ui.entropia_4.addWidget(self.canvas5)
        self.ui.densidad_3.addWidget(self.canvas6)
        
 
        #botones
        self.ui.cargar_img.clicked.connect(self.cargar_una) 
        self.ui.graficar_img.clicked.connect(self.graficadorArriba)
        self.ui.guardar_seccion.clicked.connect(self.graficadorAbajo)
        self.ui.eliminar_elemento.clicked.connect(self.eliminar_item)
        self.ui.eliminar.clicked.connect(self.eliminar_seccion)
        self.ui.name_arriba.setText("Mamografía a analizar")
        self.ui.preprocesar.clicked.connect(lambda: self.preprocesamiento(self.ui.lista_img))
        self.ui.lista_seccion.clicked.connect(self.analizar_roi)
        self.analizar_lista=0
        self.ui.lista_img.clicked.connect(self.analizar_imagen)
        self.ui.analizar.clicked.connect(lambda: self.analisis(self.analizar_lista))
        self.ui.borrar_1.clicked.connect(self.borrar_analisis1)
        self.ui.borrar_2.clicked.connect(self.borrar_analisis2)
        self.indicador=0


        self.ui.sleccionar.clicked.connect(self.seleccion_roi)

        #self.canvas.mousePressEvent = self.mousePressEvent1
        #self.canvas.mouseMoveEvent = self.mouseMoveEvent1
        #self.canvas.mouseReleaseEvent = self.mouseReleaseEvent1
        
        #self.begin, self.destination = QPoint(), QPoint()

        #Base de Datos
        self.marcos = []
        self.rois = []
        self.Qlistrois = []
        
        self.show()

    def analizar_roi(self):
        self.analizar_lista=1
        
    def analizar_imagen(self):
        self.analizar_lista=0

    def analisis(self,flag):

        self.a2.clear() #borra lo que habia 
        self.a3.clear() #borra lo que habia antes


        if flag==0: #voy a analizar la imagen completa
            item_index = self.ui.lista_img.currentRow() 
            img=self.ui.lista_img.item(item_index).data(QtCore.Qt.UserRole)

            #Densidad
            density=self.calcula_densidad(img)
            self.a2.imshow(density,cmap='gray')
            self.canvas3.draw()

            #Texturas
            dis,cor,hom,ene,cont,entrop = self.texturas(img)
            
            self.ui.disimilitud.setText(str(np.round((dis),2)))
            self.ui.correlacion.setText(str(np.round((cor),2)))
            self.ui.homogen.setText(str(np.round((hom),2)))
            self.ui.energia.setText(str(np.round((ene),2)))
            self.ui.contraste.setText(str(np.round((cont),2)))
            self.ui.entropia.setText(str(np.round((entrop),2)))

            #Grafico entropia

            entropy_graf=self.graficador_entropia(img)
            self.a3.imshow(entropy_graf,cmap='viridis')
        
            self.canvas4.draw()

            #ML: Random Forest
            
            path2= QtWidgets.QFileDialog.getOpenFileName(None,'Elegir modelo entrenado ',' C:\\','*.joblib' ) 
            
            nom_modelo=(path2[0].split("/")[-1])
            self.ui.modelo_cargado.setText(nom_modelo)
            clf_RF=joblib.load(path2[0])
            new_in = [[dis  ,   cor  ,    hom,  ene , cont,  entrop]]
            new_out = clf_RF.predict(new_in)
            self.ui.prediccion.setText(new_out[0])

        else:

            self.a4.clear()
            self.a5.clear() 

            item_index = self.ui.lista_seccion.currentRow() 
            roi=self.ui.lista_seccion.item(item_index).data(QtCore.Qt.UserRole)

            #Densidad
            density=self.calcula_densidad(roi)
            self.a5.imshow(density,cmap='gray')
            self.canvas6.draw()

            #Texturas
            dis,cor,hom,ene,cont,entrop = self.texturas(roi)
            
            self.ui.disimilitud_3.setText(str(np.round((dis),2)))
            self.ui.correlacion_3.setText(str(np.round((cor),2)))
            self.ui.homogen_3.setText(str(np.round((hom),2)))
            self.ui.energia_3.setText(str(np.round((ene),2)))
            self.ui.contraste_3.setText(str(np.round((cont),2)))
            self.ui.entropia_3.setText(str(np.round((entrop),2)))

            #Grafico entropia

            entropy_graf=self.graficador_entropia(roi)
            self.a4.imshow(entropy_graf,cmap='viridis')

            self.canvas5.draw()

        

    def texturas(self,img):
        
        glcm = graycomatrix(img, [3], [0], 256, symmetric=True, normed=True)
        glcm[:,:,0,0][0,:] = 0

        dis = ft.texture.graycoprops(glcm, 'dissimilarity')[0, 0]
        cor = ft.texture.graycoprops(glcm, 'correlation')[0, 0]
        hom = ft.texture.graycoprops(glcm, 'homogeneity')[0, 0]
        ene = ft.texture.graycoprops(glcm, 'energy')[0, 0]
        cont = ft.texture.graycoprops(glcm, 'contrast')[0, 0]
        entropia = shannon_entropy(img)
        
        return dis,cor,hom,ene,cont,entropia

    def graficador_entropia(self,imagen):
        entr_img = entropy(imagen,disk(5))

        return entr_img

    def histograma(self,imagen):

        bits=8
        tam=2**bits
        histograma = np.zeros(tam)
        f,c=len(imagen),len(imagen[0])
                
        #Recorro la imagen píxel a píxel 
        for i in range (0,f):
            for j in range (0,c):
                a=int(imagen[i,j])
                histograma[a]+=1
        
        return(histograma)


    def calcula_densidad(self,imagen):
        M,N = imagen.shape
        pixel_vals = imagen.reshape((-1)) 
        pixel_vals = np.float32(pixel_vals) #el algortimo de cv2.kmeans nos pide este tipo de dato
        # Definir criterio de corte = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.5)
        # Definir centroides
        flags = cv2.KMEANS_RANDOM_CENTERS
        # Aplicamos kmeans

        cluster_i = np.zeros((M,N)) 

        for i in range (4,8):
            #print(i)
            compactness,labels,centers = cv2.kmeans(pixel_vals,i,None,criteria,100,flags)
            center = np.uint8(centers)
            img_kmeans = center[labels.flatten()]
            img_kmeans = img_kmeans.reshape((imagen.shape))

            #hist = plt.hist(np.ravel(img_kmeans),bins=256, range=(0,255))
            hist=[self.histograma(img_kmeans)]

            #plt.close()
            #plt.show()
            #print(hist)
            val_clusters = []
            for i in range(len(hist[0])):

                valores=hist[0]
        
                if valores[i] > 0:
                    curr = np.where(valores == valores[i])
                    #print(curr[0][0])
                    val_clusters.append(curr[0][0])

            cluster_i = (img_kmeans == val_clusters[len(val_clusters)-1] )

            glcm = graycomatrix(cluster_i, [3], [0], 256, symmetric=True, normed=True)
            glcm[:,:,0,0][0,:] = 0

            hom = ft.texture.graycoprops(glcm, 'homogeneity')[0, 0]
            ene = ft.texture.graycoprops(glcm, 'energy')[0, 0]
            cont = ft.texture.graycoprops(glcm, 'contrast')[0, 0]

            #print(hom,ene,cont,val_clusters)

            if hom > 0.9 and ene > 0.6 and cont < 0.2 :
                return cluster_i

        return cluster_i

    def borrar_analisis1(self):

        self.ui.disimilitud.clear()
        self.ui.correlacion.clear()
        self.ui.homogen.clear()
        self.ui.energia.clear()
        self.ui.contraste.clear()
        self.ui.entropia.clear()
        self.ui.prediccion.clear()
        self.ui.modelo_cargado.clear()

        self.densidad.clear() #borra lo que habia antes
        self.a2 = self.densidad.add_subplot(111)
        self.canvas3.draw()

        self.entropia_1.clear() #borra lo que habia antes
        self.a3 = self.entropia_1.add_subplot(111)
        self.canvas4.draw()


    def borrar_analisis2(self):

        self.ui.disimilitud_3.clear()
        self.ui.correlacion_3.clear()
        self.ui.homogen_3.clear()
        self.ui.energia_3.clear()
        self.ui.contraste_3.clear()
        self.ui.entropia_3.clear()

        self.densidad_3.clear() #borra lo que habia antes
        self.a5 = self.densidad_3.add_subplot(111)
        self.canvas6.draw()

        self.entropia_4.clear() #borra lo que habia antes
        self.a4 = self.entropia_4.add_subplot(111)
        self.canvas5.draw()


        


    def cargar_una(self): 

        path= QtWidgets.QFileDialog.getOpenFileName(None,'Examinar ordenador',' C:\\' ,'Image Files(*.png *.jpg *.bmp)') 
        img = sitk.ReadImage(path[0])
        img = sitk.GetArrayFromImage(img)[:,:,0]

        if img.shape[0]>img.shape[1]:
            img = img.T
        nombre_img=(path[0].split("/")[-1])
        img=self.padding(img)
               
        item_nuevo = QtWidgets.QListWidgetItem()
        item_nuevo.setText(nombre_img)
        item_nuevo.setData(QtCore.Qt.UserRole, img)

#Agrego un casillero a las bases de datos
        self.marcos.append([])
        self.rois.append([])
        self.Qlistrois.append([])

        self.ui.lista_img.addItem(item_nuevo)

    def preprocesamiento(self,lista):

        item_index = lista.currentRow() 

        if item_index != None:
            
            if self.ui.sin_fondo.isChecked():
                nombre= lista.item(item_index).text()
                lado=nombre.split("_")[1] #si es mama derecha R o izq L 
                img=lista.item(item_index).data(QtCore.Qt.UserRole)
                img_nueva=img.copy()
                img_nueva=self.eliminar_fondo(img_nueva,lado)
                nombre_nuevo=nombre.split(".")[0] + str('_sf_.') + nombre.split(".")[1]
               

            if self.ui.filtro.isChecked():

                nombre= lista.item(item_index).text()
                img=lista.item(item_index).data(QtCore.Qt.UserRole)
                img_nueva=img.copy()
                img_nueva = scipy.signal.medfilt2d(img, kernel_size=3)
                nombre_nuevo=nombre.split(".")[0] + str('_filt_.') + nombre.split(".")[1]

            #Agregamos a la lista la imagen preprocesada

            item_nuevo = QtWidgets.QListWidgetItem()
            item_nuevo.setText(nombre_nuevo)
            item_nuevo.setData(QtCore.Qt.UserRole, img_nueva)

    #Agrego un casillero a las bases de datos
            self.marcos.append([])
            self.rois.append([])
            self.Qlistrois.append([])

            lista.addItem(item_nuevo)               

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Debe seleccionar primero una imagen de la lista.")
            msg.setWindowTitle("Error")
            msg.exec_()

    def padding(self,imagen):

        d = imagen.shape

        pf = 3
        pc = 5

        marco = np.ones((pf,pc))

        fil = d[0]
        col = d[1]

        ratfil = fil/pf
        ratcol = col/pc

        if fil/pf > col/pc:
            print('hay que agregar columnas')
            pad = np.zeros((fil,fil*pc//(pf)))
            ofs = (pad.shape[1]-col)//2
            pad[0:imagen.shape[0],ofs:imagen.shape[1]+ofs]= np.nan
            pad=imagen
            

        else:
            print('hay que agregar filas')
            pad = np.zeros((col*pf//(pc),col))
            ofs = (pad.shape[0]-fil)//2
            pad[ofs:imagen.shape[0]+ofs,0:imagen.shape[1]]= np.nan
            pad=imagen
            
        return pad

    def eliminar_fondo(self,image,lado):

        background = np.asarray(image)

        if lado =='L': #mama izquierda
            seed = (1,image.shape[1]-2)
            seedValue = image[seed[0],seed[1]]

        if lado=='R':  #mama derecha
            seed = (1,1)
            seedValue = image[seed[0],seed[1]]
        
        seedValue = background[seed]
        vmin = int(seedValue-25)
        vmax = int(seedValue+25)

        regionGrowingFilter = sitk.ConnectedThresholdImageFilter()
        regionGrowingFilter.AddSeed(seed)
        regionGrowingFilter.SetLower(vmin)
        regionGrowingFilter.SetUpper(vmax)
        regionGrowingFilter.SetReplaceValue(255)
        background = sitk.GetArrayFromImage(regionGrowingFilter.Execute(sitk.GetImageFromArray(background)))
        background = cv2.bitwise_not(background)

        for row in range(0,background.shape[0]):
            for col in range(0,background.shape[1]):
                if background[row,col] == 255:
                    background[row,col] = 1
                else:
                    background[row,col] = 0

        # Math morphology to delete non-connected pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(80,80))
        background = cv2.erode(background, kernel, iterations = 1)
        background = cv2.dilate(background, kernel, iterations = 1)

        # Improve breast mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
        background = cv2.erode(background, kernel, iterations = 3)
        imagen_sin_fondo=image-background

        return imagen_sin_fondo


    def seleccionar_img(self, texto):

        item = self.ui.lista_img.currentItem()

        if item != None:
            
            nombre = item.text()
            texto.setText(nombre)
            
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Debe seleccionar primero una imagen de la lista.")
            msg.setWindowTitle("Error")
            msg.exec_()

    def graficadorArriba(self):

        cant_img = self.ui.lista_img.count()

        if cant_img != 0:  

            self.grafico_a.clear() #borra lo que habia antes
            self.a1 = self.grafico_a.add_subplot(111)
            
            item_index = self.ui.lista_img.currentRow() 
        
            imagen = self.ui.lista_img.item(item_index).data(QtCore.Qt.UserRole)
            #imagen=self.padding(imagen)

            self.a1.imshow(imagen,cmap='gray')

        #Pongo los Marcos Existentes
            for i in range(len(self.marcos[item_index])):
                newax = self.a1.twinx()
                newax.imshow(self.marcos[item_index][i],cmap='gray',vmin=0,vmax=255)
                
            self.canvas.draw()

        #Reseteo lista de ROIs
            self.limpiarListaROI()
            
            for num in range(len(self.rois[item_index])):
                nombre_roi=(f'roi_{num}')
                item_nuevo = QtWidgets.QListWidgetItem()
                item_nuevo.setText(nombre_roi)
                item_nuevo.setData(QtCore.Qt.UserRole, self.rois[item_index][num])
                self.ui.lista_seccion.addItem(item_nuevo)

        else:
            self.grafico_a.clear() #borra lo que habia antes
            self.a1 = self.grafico_a.add_subplot(111)
            self.canvas.draw()
            #msg = QMessageBox()
            #msg.setIcon(QMessageBox.Critical)
            #msg.setText("Debe seleccionar una imagen de la lista.")
            #msg.setWindowTitle("Error")
            #msg.exec_()

    
    def limpiarListaROI(self):
        while self.ui.lista_seccion.count() != 0:
                self.ui.lista_seccion.takeItem(0)

    def eliminar_item(self):

        item_index = self.ui.lista_img.currentRow() 
        item = self.ui.lista_img.currentItem()
        if item != None:
            delete(item)
            self.marcos.pop(item_index)
            self.rois.pop(item_index)
            self.Qlistrois.pop(item_index)

            self.limpiarListaROI()
            
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Debe seleccionar primero una imagen de la lista.")
            msg.setWindowTitle("Error")
            msg.exec_()

    def mousePressEvent1(self, event):
        if event.buttons() & Qt.LeftButton:
            self.begin = event.pos()
            self.destination = self.begin
            print(self.begin, self.destination)

    def seleccion_roi(self):

        if self.indicador==0:

            self.canvas.mousePressEvent = self.mousePressEvent1
            self.canvas.mouseMoveEvent = self.mouseMoveEvent1
            self.canvas.mouseReleaseEvent = self.mouseReleaseEvent1
            
            self.begin, self.destination = QPoint(), QPoint() 
            self.indicador=1 

        else:

            self.canvas.mouseMoveEvent = self.mouseMoveEvent2
            self.canvas.mouseReleaseEvent = self.mouseReleaseEvent2
            
            self.begin, self.destination = QPoint(), QPoint() 
            self.indicador=0

    def mouseMoveEvent2(self, event):
        pass

    def mouseReleaseEvent2(self, event):
        pass


    def mouseMoveEvent1(self, event):
        if event.buttons() & Qt.LeftButton:		
            self.destination = event.pos()
            print(self.begin, self.destination)

    def mouseReleaseEvent1(self, event):
        if event.button() & Qt.LeftButton:
           self.crearRoi()
           self.begin, self.destination = QPoint(), QPoint()
            
    def crearRoi(self):
        item_index = self.ui.lista_img.currentRow() 
        imagen = self.ui.lista_img.item(item_index).data(QtCore.Qt.UserRole)

        print('--------------------------------------------')
        print(imagen.shape)
        (start,end) = self.escalado(imagen)
        print(self.escalado(imagen))
        box = self.selecBox(imagen,start,end)
        roi = np.copy(imagen[start[0]:end[0],start[1]:end[1]])

        self.marcos[item_index].append(box)
        self.rois[item_index].append(roi)
        self.agregarRoi(roi)

        newax = self.a1.twinx()
        newax.imshow(box,cmap='gray',vmin=0,vmax=255)

        self.canvas.draw()

        print(self.begin, self.destination)
        
    def selecBox(self, imagen, s, e):
        g = imagen.shape[1]//100
        box = np.empty(imagen.shape)
        box[:] = np.nan
        box[s[0]:e[0],s[1]:e[1]] = 255
        box[s[0]+g:e[0]-g,s[1]+g:e[1]-g] = np.nan

        return box

    def escalado(self, imagen):
        starty = self.begin.y()
        startx = self.begin.x()
        endy = self.destination.y()
        endx = self.destination.x()

        starty = ((starty-30)*imagen.shape[0])//190
        startx = ((startx-60)*imagen.shape[1])//330
        endy = ((endy-30)*imagen.shape[0])//190
        endx = ((endx-60)*imagen.shape[1])//330

        return ((starty,startx),(endy,endx))
 
    def agregarRoi(self,roi):
        id_imagen = self.ui.lista_img.currentRow()
        num = len(self.rois[id_imagen])
        nombre_roi=(f'roi_{num}')
        item_nuevo = QtWidgets.QListWidgetItem()
        item_nuevo.setText(nombre_roi)
        item_nuevo.setData(QtCore.Qt.UserRole, roi)
        self.ui.lista_seccion.addItem(item_nuevo)
        self.Qlistrois[id_imagen].append(item_nuevo)

    def graficadorAbajo(self):
        cant_img = self.ui.lista_seccion.count()

        if cant_img != None:  

            self.grafico_b.clear() #borra lo que habia antes
            self.b1 = self.grafico_b.add_subplot(111)
            
            item_index = self.ui.lista_seccion.currentRow() 
        
            imagen = self.ui.lista_seccion.item(item_index).data(QtCore.Qt.UserRole)

            self.b1.imshow(imagen,cmap='gray')

            self.canvas2.draw()





    def eliminar_seccion(self):
        item_index = self.ui.lista_img.currentRow() 
        roi_index = self.ui.lista_seccion.currentRow() 
        item = self.ui.lista_seccion.currentItem()
        if item != None:
            delete(item)
            self.marcos[item_index].pop(roi_index)
            self.rois[item_index].pop(roi_index)
            self.Qlistrois[item_index].pop(roi_index)
            self.graficadorArriba()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(15, 15, 15))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(83,255,255))#.lighter())
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtCore.Qt.darkGray)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtCore.Qt.darkGray)
    app.setPalette(palette)
    ventana = Aplicacion()
    ventana.showMaximized()
    ventana.show
    sys.exit(app.exec_())