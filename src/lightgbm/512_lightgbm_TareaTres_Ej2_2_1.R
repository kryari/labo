# LightGBM  cambiando algunos de los parametros

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("lightgbm")

#Aqui se debe poner la carpeta de la computadora local
setwd("C:/Users/ARI/Desktop/ITBA/5.Mineria/")   #Establezco el Working Directory

#cargo el dataset donde voy a entrenar
dataset  <- fread("./datasets/paquete_premium_202011.csv", stringsAsFactors= TRUE)


#paso la clase a binaria que tome valores {0,1}  enteros
dataset[ , clase01 := ifelse( clase_ternaria=="BAJA+2", 1L, 0L) ]

dataset[ , ctarjeta_visa_trx2 := ctarjeta_visa_trx ]
dataset[ ctarjeta_visa==0 & ctarjeta_visa_trx==0,  ctarjeta_visa_trx2 := NA ]

#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )


#dejo los datos en el formato que necesita LightGBM
dtrain  <- lgb.Dataset( data= data.matrix(  dataset[ , campos_buenos, with=FALSE]),
                        label= dataset$clase01 )

#genero el modelo con los parametros por default
modelo  <- lgb.train( data= dtrain,
                      param= list( objective=        "binary",
                                   metric= "custom",
                                   first_metric_only= TRUE,
                                   boost_from_average= TRUE,
                                   feature_pre_filter= FALSE,
                                   verbosity= -100,
                                   seed= 999983 ,
                                   max_depth=-1,
                                   min_gain_to_split=0,
                                   lambda_l1= 0.0,         #por ahora, lo dejo fijo
                                   lambda_l2= 0.0,         #por ahora, lo dejo fijo
                                   max_bin=31,
                                   num_iterations=     166,
                                   force_row_wise= TRUE,
                                   learning_rate=0.0395368223132189,
                                   feature_fraction=    0.659740772685352,
                                   min_data_in_leaf= 4840,
                                   num_leaves=         691,
                                   prob_corte = 0.0164003733866544

                                   )    #para que los alumnos no se atemoricen con tantos warning)
                      )  

#aplico el modelo a los datos sin clase
dapply  <- fread("./datasets/paquete_premium_202101.csv")

dapply[ , ctarjeta_visa_trx2 := ctarjeta_visa_trx ]
dapply[ ctarjeta_visa==0 & ctarjeta_visa_trx==0,  ctarjeta_visa_trx2 := NA ]


#aplico el modelo a los datos nuevos
prediccion  <- predict( modelo, 
                        data.matrix( dapply[, campos_buenos, with=FALSE ]) )


#Genero la entrega para Kaggle
entrega  <- as.data.table( list( "numero_de_cliente"= dapply[  , numero_de_cliente],
                                 "Predicted"= as.integer(prediccion > 1/60 ) )  ) #genero la salida

dir.create( "./labo/exp/",  showWarnings = FALSE ) 
dir.create( "./labo/exp/KA2512/", showWarnings = FALSE )
archivo_salida  <- "./labo/exp/KA2512/KA_512_001_TareaTres_Ej2_2_1.csv"

#genero el archivo para Kaggle
fwrite( entrega, 
        file= archivo_salida, 
        sep= "," )


#ahora imprimo la importancia de variables
tb_importancia  <-  as.data.table( lgb.importance(modelo) ) 
archivo_importancia  <- "./labo/exp/KA2512/512_importancia_001_TareaTres_Ej2_2_1.txt"

fwrite( tb_importancia, 
        file= archivo_importancia, 
        sep= "\t" )

