# elderReact

## Estrazione feature
Per effettuare l'estrazione delle feature bisogna eseguire lo script shell "feature_extraction_1.sh".
Lo script eseguira i file python presenti nel repository. L'esecuzione potrà durare anche un giorno.

Una volta eseguito il file "feature_extraction_1.sh" bisogna eseguire il file python "my_demo_FSANET_sdd.py" presente in "FSA-Net/demo"; per fare questo leggere
le dependencies sul GitHub di FSA-Net [qui](https://github.com/shamangary/FSA-Net/blob/master/README.md). L'esecuzione di questo
script può durare anche mezza giornata.

Dopo aver eseguito correttamente "my_demo_FSANET_sdd.py" eseguire "feature_extraction_2.sh", la durata di questo script sarà di qualche secondo.

Una volta eseguito l'ultimo script il dataset sarà stato generato correttamente.

**Per far funzionare lo script bisogna modificare il path assoluto in riga 15 del file "data_extraction.py"**.

## Addestramento dei modelli di ML
Per effettuare l'addestramento dei modelli di machine learning basta eseguire il file "run_models.py".
L'addestramento durerà circa un giorno.
