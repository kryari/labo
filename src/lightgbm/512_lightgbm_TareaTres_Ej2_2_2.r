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
                                   num_iterations=     9999,
                                   num_leaves=         31,
                                   feature_fraction=    1.0,
                                   min_data_in_leaf= 20,
                                   seed= 999983,
                                   prob_corte =       1/60,
                                   learning_rate =      0.1,
                                   max_depth=  -1,         # -1 significa no limitar,  por ahora lo dejo fijo
                                   min_gain_to_split= 0.0, #por ahora, lo dejo fijo
                                   lambda_l1= 0.0,         #por ahora, lo dejo fijo
                                   lambda_l2= 0.0,
                                   verbosity= -100,
                                   force_row_wise= TRUE
                                   )
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
archivo_salida  <- "./labo/exp/KA2512/KA_512_001_TareaTres_Ej2_2_2.csv"

#genero el archivo para Kaggle
fwrite( entrega, 
        file= archivo_salida, 
        sep= "," )


#ahora imprimo la importancia de variables
tb_importancia  <-  as.data.table( lgb.importance(modelo) ) 
archivo_importancia  <- "./labo/exp/KA2512/512_importancia_001_TareaTres_Ej2_2_2.txt"

fwrite( tb_importancia, 
        file= archivo_importancia, 
        sep= "\t" )
