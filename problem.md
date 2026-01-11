porblem je s doplnovanim nových dat 

Techmania chce vše dávat  do excelu takže potřebuji pipline s PATCH na detekci změny v záznamu 
Ten excel bude vycházet z toho template který je nastave pro rok 2026 a má tam default hodnoty pro predikci 

tentrénovací techmania_with_weather_and_holidays.csv sebude používat pouze pro tréning 

kopie všech záznamů by měla být v  SQLite a z ní pak budou vycházet  grafy a statistiky 

v SQlite by měla být Historická tabulka  s  hodnotami ztechmania_with_weather_and_holidays.csv 
které se pak z té db budeou dále používat na frontendu 
pak by tam měla být tabulka prediction s tím že ta bude pouze v db a bude verzovana na jednotlivé datumi takže jeden den může mít vícero predikcí (pokud  jsou rozdílné) 
v SQLite by měla být taky tabulka pro hodnoty z template.csv  které budou sloužit pro data pro dělaní predikcí pro rok 2026 

Pro detekci změn v tom Excelu musí být jiná Pipline která když detekuje změnu že se třeba přidali všechny hodnoty pro den 1.1.2026 (takže záznam z  template.csv který bude v SQLite) tak se automaticky musí zaznamenat v SQLite a musí u nich být flag nebo něco že se mohou použít na grafy atd... (v podstate to znamená že se do těch polí přidali reálná data o návštevnosti počasí atd... takže se musí flagnout že ten nový záznam co nebyl v historických datech je compleate  )

také by se mělo zakázat dělat predikce do minulosti protože to je blbost takže by se to mělo řídit podle datumu 