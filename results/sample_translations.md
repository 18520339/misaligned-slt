# Sample Translation Comparison


## Head Truncation (20%)

### Sample: `test/03July_2011_Sunday_tagesschau-1665`

| | Text |
|---|---|
| **Reference** | in den übrigen landesteilen wird es meist freundlich dank einer hochdruckzone die von der biskaya bis zu den shetlandinseln reicht . |
| **Clean output** | sonst wird es dank hochdruckeinfluss freundlicher und das hoch verlagert sich bis in die mitte . |
| **Misaligned output** | es wird wieder freundlicher und das hoch verlagert sich langsam bis in die mitte . |
| **Failure type** | acceptable |

### Sample: `test/12July_2010_Monday_tagesschau-374`

| | Text |
|---|---|
| **Reference** | morgen gibt es im osten und südosten bei einer mischung aus sonne und wolken zum teil kräftige schauer oder gewitter . |
| **Clean output** | morgen im osten und südosten eine mischung aus sonne und wolken teilweise gibt es kräftige schauer oder gewitter . |
| **Misaligned output** | in der südosthälfte eine mischung aus sonne und wolken mit teilweise kräftigen schauern und gewittern . |
| **Failure type** | acceptable |

### Sample: `test/02December_2009_Wednesday_tagesschau-4039`

| | Text |
|---|---|
| **Reference** | und nun die wettervorhersage für morgen donnerstag den dritten dezember . |
| **Clean output** | und nun die wettervorhersage für morgen donnerstag den dritten dezember . |
| **Misaligned output** | und nun die wettervorhersage für morgen donnerstag den dritten dezember . |
| **Failure type** | acceptable |

### Sample: `test/11November_2010_Thursday_tagesschau-3564`

| | Text |
|---|---|
| **Reference** | teilweise kräftiger regen zieht heute nacht über uns ostwärts . |
| **Clean output** | die teilweise kräftigen regenfälle breiten sich heute nacht von westen aus . |
| **Misaligned output** | die regenwolken breiten sich im tagesverlauf richtung osten aus und erreichen am abend auch den westen deutschlands . |
| **Failure type** | hallucination |

### Sample: `test/27August_2009_Thursday_tagesschau-3267`

| | Text |
|---|---|
| **Reference** | und nun die wettervorhersage für morgen freitag den achtundzwanzigsten august . |
| **Clean output** | und nun die wettervorhersage für morgen freitag den achtundzwanzigsten august . |
| **Misaligned output** | und nun die wettervorhersage für morgen freitag den achtundzwanzigsten august . |
| **Failure type** | acceptable |


## Tail Truncation (20%)

### Sample: `test/24August_2010_Tuesday_tagesschau-2196`

| | Text |
|---|---|
| **Reference** | heute nacht gibt es an den küsten schauer im süden ebenfalls aber dort lassen sie allmählich nach . |
| **Clean output** | heute nacht regnet es an den küsten anfangs auch im süden hier und da sonst lässt der regen auch langsam nach . |
| **Misaligned output** | heute nacht regnet es an den küsten hier und da etwas auch im süden . |
| **Failure type** | acceptable |

### Sample: `test/08February_2010_Monday_heute-1498`

| | Text |
|---|---|
| **Reference** | und im westen haben wir kaum noch schnee aber es gibt eben auch ausnahmen die mittelgebirge . |
| **Clean output** | in der westhälfte kaum noch schnee und die schneefallgrenze sinkt weiter in den nächsten tagen aber auf den gefrierpunkt in den mittelgebirgen . |
| **Misaligned output** | in der westhälfte kaum noch schnee von südwesten lockert es zunehmend auf aber es gibt auch noch wolkenlücken . |
| **Failure type** | acceptable |

### Sample: `test/25November_2009_Wednesday_tagesschau-7677`

| | Text |
|---|---|
| **Reference** | am freitag wechselhaftes schauerwetter . |
| **Clean output** | am freitag regenschauer und gewitter im wechsel . |
| **Misaligned output** | am freitag regenschauer und gewitter im wechsel . |
| **Failure type** | acceptable |

### Sample: `test/02August_2010_Monday_heute-1243`

| | Text |
|---|---|
| **Reference** | milder bleibt es unter den regenwolken im osten mit fünfzehn sechzehn grad . |
| **Clean output** | und der regen hier an der ostsee naja da ist es nochmal mild mit fünfzehn bis sechzehn grad . |
| **Misaligned output** | und wir haben noch regen im osten und im süden da bleibt es am längsten mild mit fünfzehn bis fünfzehn grad . |
| **Failure type** | acceptable |

### Sample: `test/29March_2010_Monday_tagesschau-8387`

| | Text |
|---|---|
| **Reference** | sie bringen kräftige schauer und gewitter . |
| **Clean output** | dabei breiten sich zum teil kräftige schauer und gewitter aus . |
| **Misaligned output** | und da an dieser grenze entstehen immer wieder diese regenschauer und gewitter . |
| **Failure type** | acceptable |


## Head Contamination (20%)

### Sample: `test/16December_2009_Wednesday_tagesschau-5907`

| | Text |
|---|---|
| **Reference** | und nun die wettervorhersage für morgen donnerstag den siebzehnten dezember . |
| **Clean output** | und nun die wettervorhersage für morgen donnerstag den siebzehnten dezember . |
| **Misaligned output** | und nun die wettervorhersage für morgen donnerstag den siebzehnten dezember . |
| **Failure type** | acceptable |

### Sample: `test/30January_2013_Wednesday_tagesschau-476`

| | Text |
|---|---|
| **Reference** | auch am samstag regnet oder schneit es verbreitet später wird es im westen und norden freundlicher . |
| **Clean output** | auch am samstag regnet oder schneit es verbreitet später wird es im westen und norden freundlicher . |
| **Misaligned output** | bis zu zwanzig grad auch am samstag regnet oder schneit es zeitweise später wird es im westen und norden freundlicher . |
| **Failure type** | acceptable |

### Sample: `test/01October_2009_Thursday_tagesschau-421`

| | Text |
|---|---|
| **Reference** | im süden freundliches wetter . |
| **Clean output** | richtung süden ist es freundlicher . |
| **Misaligned output** | kühler bleibt es noch im süden . |
| **Failure type** | acceptable |

### Sample: `test/18April_2011_Monday_tagesschau-1314`

| | Text |
|---|---|
| **Reference** | und nun die wettervorhersage für morgen dienstag den neunzehnten april . |
| **Clean output** | und nun die wettervorhersage für morgen dienstag den neunzehnten april . |
| **Misaligned output** | und nun die wettervorhersage für morgen dienstag den neunzehnten april . |
| **Failure type** | acceptable |

### Sample: `test/14August_2011_Sunday_tagesschau-3230`

| | Text |
|---|---|
| **Reference** | dabei kommt es zu heftigen schauern und gewittern die teilweise unwetterartig sein können . |
| **Clean output** | dabei gibt es zum teil kräftige schauer und gewitter die örtlich auch unwetterartig sein können . |
| **Misaligned output** | im laufe des tages entwickeln sich mitunter kräftige schauer und gewitter die örtlich auch unwetterartig sein können . |
| **Failure type** | acceptable |


## Tail Contamination (20%)

### Sample: `test/05May_2010_Wednesday_tagesschau-3360`

| | Text |
|---|---|
| **Reference** | am wochenende wird es dann etwas wärmer das ganze bei wechselhaftem wetter mit schauern gewittern und gelegentlichem sonnenschein . |
| **Clean output** | im laufe des tages wird es immer wärmer dabei unbeständiger es gibt gewitter aber auch immer wieder sonnenschein . |
| **Misaligned output** | sonst wird es wechselhaft mit schauern und gewittern die uns auch morgen wieder eine mischung aus sonne und wolken bringen . |
| **Failure type** | acceptable |

### Sample: `test/17January_2011_Monday_tagesschau-7005`

| | Text |
|---|---|
| **Reference** | auch im süden gibt es eine menge wolken oder nebel aber regen fällt dort nur selten . |
| **Clean output** | auch im süden gibt es viele wolken oder nebelfelder aber regen fällt dort kaum . |
| **Misaligned output** | auch im süden gibt es viele wolken oder nebelfelder aber regen fällt dort kaum noch vor allem südlich der donau am bodensee . |
| **Failure type** | acceptable |

### Sample: `test/04January_2010_Monday_tagesschau-8550`

| | Text |
|---|---|
| **Reference** | im norden mäßiger wind an den küsten weht er teilweise frisch . |
| **Clean output** | im norden mäßiger bis frischer an der küste zeitweise starker wind . |
| **Misaligned output** | der wind weht im norden mäßig bis frisch an den küsten stark mit stürmischen böen im süden vereinzelt noch sturmböen . |
| **Failure type** | acceptable |

### Sample: `test/17April_2010_Saturday_tagesschau-399`

| | Text |
|---|---|
| **Reference** | die luft die in der kommenden woche zu uns strömt ist deutlich kühler . |
| **Clean output** | in der neuen woche strömt dann noch kühlere luft zu uns . |
| **Misaligned output** | in der neuen woche strömt dann noch kühlere luft zu uns nach deutschland . |
| **Failure type** | acceptable |

### Sample: `test/10December_2009_Thursday_heute-7945`

| | Text |
|---|---|
| **Reference** | teilweise im nordwesten ein bisschen schneeregen . |
| **Clean output** | im norden und nordwesten scheint die sonne ab und zu mal für längere zeit auch am samstag schneit es zeitweise im norden und westen schneit es zeitweise im nordwesten teilweise regen oder schneeregen . |
| **Misaligned output** | im norden und nordwesten scheint die sonne ab und zu mal für längere zeit auch am samstag schneit es zeitweise im norden und westen schneit es zeitweise im nordwesten gefrierender regen oder schneeregen in tiefer luftdruck . |
| **Failure type** | hallucination |


## Head Trunc + Tail Trunc


### Head=20%, Tail=20%

#### Sample: `test/05February_2010_Friday_tagesschau-2924`

| | Text |
|---|---|
| **Reference** | dabei ist es teils stark bewölkt oder neblig trüb teils freundlich mit sonnenschein . |
| **Clean output** | aber sonst ist es meist stark bewölkt oder neblig trüb teils neblig trüb teils sonnig . |
| **Misaligned output** | wolkenverhangen stellenweise bildet sich nebel sonst ist es teilweise freundlich . |
| **Failure type** | acceptable |

#### Sample: `test/25November_2009_Wednesday_tagesschau-7677`

| | Text |
|---|---|
| **Reference** | am freitag wechselhaftes schauerwetter . |
| **Clean output** | am freitag regenschauer und gewitter im wechsel . |
| **Misaligned output** | am freitag regenschauer und gewitter im wechsel . |
| **Failure type** | acceptable |

#### Sample: `test/09August_2010_Monday_heute-5890`

| | Text |
|---|---|
| **Reference** | ganz anders die trockene hitze über russland mit fünfunddreißig bis vierzig grad . |
| **Clean output** | und dort haben wir morgen auch trockenes sonniges wetter mit fünf bis vier grad . |
| **Misaligned output** | an der küste bei russland morgen trockenes wetter mit viel sonnenschein fünf bis fünf grad . |
| **Failure type** | acceptable |

#### Sample: `test/22February_2010_Monday_tagesschau-3246`

| | Text |
|---|---|
| **Reference** | im südosten ist es heute nacht trocken sonst regnet es zeitweise in der nordhälfte teilweise kräftig . |
| **Clean output** | im norden und nordosten bleibt es heute nacht meist trocken sonst regnet es zeitweise im norden und osten zum teil ergiebig . |
| **Misaligned output** | heute nacht bleibt es meist trocken aber später beginnt es im süden wieder zu regnen . |
| **Failure type** | acceptable |

#### Sample: `test/26July_2010_Monday_tagesschau-6266`

| | Text |
|---|---|
| **Reference** | am mittwoch breiten sich von nordwesten neue wolken mit regen oder schauern aus örtlich sind auch gewitter dabei . |
| **Clean output** | am mittwoch breiten sich von nordwesten schauer und gewitter aus die örtlich unwetterartig sein können . |
| **Misaligned output** | von nordwesten breiten sich später regenschauer und gewitter aus die stellenweise kräftig sein können . |
| **Failure type** | acceptable |


### Head=10%, Tail=40%

#### Sample: `test/27May_2010_Thursday_tagesschau-8382`

| | Text |
|---|---|
| **Reference** | am sonntag überall unbeständiges wetter teilweise kräftige regenfälle teilweise auch gewitter . |
| **Clean output** | am sonntag unbeständiges wetter teilweise regnet es kräftig auch gewitter sind mit dabei . |
| **Misaligned output** | am donnerstag unbeständiges wetter teilweise ist es stark bewölkt . |
| **Failure type** | acceptable |

#### Sample: `test/05September_2009_Saturday_tagesschau-4085`

| | Text |
|---|---|
| **Reference** | in der nacht an der see noch stürmische böen . |
| **Clean output** | in der nacht stürmisch auf den bergen sturmböen . |
| **Misaligned output** | in der nacht kühlt es weiter ab . |
| **Failure type** | acceptable |

#### Sample: `test/14August_2009_Friday_tagesschau-75`

| | Text |
|---|---|
| **Reference** | örtlich schauer oder gewitter die heftig sein können . |
| **Clean output** | hier und da gibt es schauer oder gewitter die heftig sein können . |
| **Misaligned output** | hier und da fallen schauer . |
| **Failure type** | acceptable |

#### Sample: `test/03February_2010_Wednesday_heute-2362`

| | Text |
|---|---|
| **Reference** | ich wünsche ihnen einen schönen abend und machen sie es gut . |
| **Clean output** | ihnen noch einen schönen abend und machen sie es gut . |
| **Misaligned output** | ihnen noch einen schönen abend und machen sie es gut . |
| **Failure type** | acceptable |

#### Sample: `test/16December_2009_Wednesday_tagesschau-5907`

| | Text |
|---|---|
| **Reference** | und nun die wettervorhersage für morgen donnerstag den siebzehnten dezember . |
| **Clean output** | und nun die wettervorhersage für morgen donnerstag den siebzehnten dezember . |
| **Misaligned output** | und nun die wettervorhersage für morgen donnerstag den vierten juli . |
| **Failure type** | acceptable |


### Head=40%, Tail=10%

#### Sample: `test/21May_2010_Friday_tagesschau-5094`

| | Text |
|---|---|
| **Reference** | abgesehen von schauer und gewitterböen weht der wind schwach bis mäßig aus nordwest bis nord . |
| **Clean output** | dazu weht ein schwacher bis mäßiger wind aus unterschiedlichen richtungen . |
| **Misaligned output** | es weht ein schwacher bis mäßiger südwestwind . |
| **Failure type** | acceptable |

#### Sample: `test/05September_2009_Saturday_tagesschau-4085`

| | Text |
|---|---|
| **Reference** | in der nacht an der see noch stürmische böen . |
| **Clean output** | in der nacht stürmisch auf den bergen sturmböen . |
| **Misaligned output** | und es wird auch stürmisch in den nächsten tagen . |
| **Failure type** | acceptable |

#### Sample: `test/03February_2010_Wednesday_heute-2355`

| | Text |
|---|---|
| **Reference** | im norden wird es dagegen noch trocken sein . |
| **Clean output** | im norden bleibt es meist trocken . |
| **Misaligned output** | im norden bleibt es meist trocken . |
| **Failure type** | acceptable |

#### Sample: `test/26August_2009_Wednesday_tagesschau-3213`

| | Text |
|---|---|
| **Reference** | und nun die wettervorhersage für morgen donnerstag den siebenundzwanzigsten august . |
| **Clean output** | und nun die wettervorhersage für morgen donnerstag den siebenundzwanzigsten august . |
| **Misaligned output** | und nun die wettervorhersage für morgen donnerstag den siebenundzwanzigsten august . |
| **Failure type** | acceptable |

#### Sample: `test/29August_2009_Saturday_tagesschau-5020`

| | Text |
|---|---|
| **Reference** | in den frühstunden kann sich vereinzelt nebel bilden . |
| **Clean output** | morgen früh bildet sich stellenweise nebel . |
| **Misaligned output** | örtlich bildet sich nebel . |
| **Failure type** | acceptable |


## Head Trunc + Tail Contam


### Head=20%, Tail=20%

#### Sample: `test/08February_2010_Monday_heute-1498`

| | Text |
|---|---|
| **Reference** | und im westen haben wir kaum noch schnee aber es gibt eben auch ausnahmen die mittelgebirge . |
| **Clean output** | in der westhälfte kaum noch schnee und die schneefallgrenze sinkt weiter in den nächsten tagen aber auf den gefrierpunkt in den mittelgebirgen . |
| **Misaligned output** | das schneefallgebiet breitet sich weiter richtung mitte aus . |
| **Failure type** | hallucination |

#### Sample: `test/02October_2010_Saturday_tagesschau-1300`

| | Text |
|---|---|
| **Reference** | sehr milde luft strömt zu uns . |
| **Clean output** | dabei strömt sehr milde luft zu uns . |
| **Misaligned output** | und es weht nur ein schwacher bis mäßiger südwestwind dazu . |
| **Failure type** | hallucination |

#### Sample: `test/17May_2010_Monday_heute-5576`

| | Text |
|---|---|
| **Reference** | auch noch verhältnismäßig kühl aber dann sieht es schon nach ein paar sonnigeren und auch deutlich wärmeren tagen aus . |
| **Clean output** | und es wird auch entsprechend kühl aber dann kann sich zumindest in den nächsten tagen erstmal besser die sonne zeigen und es wird auch wieder wärmer . |
| **Misaligned output** | aber dann wird es besser werden und die sonne kommt raus und die wärme kommt raus und damit schönen guten abend . |
| **Failure type** | acceptable |

#### Sample: `test/26October_2009_Monday_tagesschau-1403`

| | Text |
|---|---|
| **Reference** | am freitag scheint abseits der nebelgebiete häufig die sonne . |
| **Clean output** | am freitag nach auflösung von teilweise zähen nebelfeldern viel sonne . |
| **Misaligned output** | am freitag abseits des nebels viel sonne in der nordhälfte . |
| **Failure type** | acceptable |

#### Sample: `test/23August_2010_Monday_heute-5328`

| | Text |
|---|---|
| **Reference** | ja und so bleibt es dann auch erstmal am mittwoch . |
| **Clean output** | auch am mittwoch ändert sich noch wenig an diesem wetter . |
| **Misaligned output** | und so geht es eigentlich auch weiter erst am mittwoch mittwoch . |
| **Failure type** | acceptable |


### Head=10%, Tail=40%

#### Sample: `test/17July_2009_Friday_tagesschau-5104`

| | Text |
|---|---|
| **Reference** | in der dahinter einfließenden kaltluft gestaltet sich das wetter morgen wechselhaft . |
| **Clean output** | dabei gelangt von nordwesten kältere luft nach deutschland und damit setzt sich das wechselhafte wetter fort . |
| **Misaligned output** | die kaltluft die zu uns strömt bestimmt weiterhin unser wetter und auch das wetter in der osthälfte und im süden deutschlands heute nacht . |
| **Failure type** | acceptable |

#### Sample: `test/04January_2010_Monday_tagesschau-8552`

| | Text |
|---|---|
| **Reference** | heute nacht minus zwei grad an der nordsee und bis minus zwanzig grad in einzelnen alpentälern . |
| **Clean output** | heute nacht minus zwei grad am niederrhein und minus zwei grad an den alpen . |
| **Misaligned output** | heute nacht minus zwei grad bis minus zwei grad an den alpen und stellenweise minus ein grad im westen . |
| **Failure type** | acceptable |

#### Sample: `test/24March_2011_Thursday_tagesschau-3937`

| | Text |
|---|---|
| **Reference** | in der nacht werden die wolken im norden und nordosten dichter erste tropfen können an der ostsee fallen . |
| **Clean output** | in der nacht ist es im norden und nordosten wolkig an der ostsee kann es etwas regnen . |
| **Misaligned output** | im norden und nordosten tauchen gelegentlich auch mal dichtere wolken auf an der ostsee kann es hier und da etwas regnen sonst meist bewölkt . |
| **Failure type** | acceptable |

#### Sample: `test/09July_2010_Friday_tagesschau-591`

| | Text |
|---|---|
| **Reference** | zunächst ist das gewitterrisiko nur im westen erhöht von tag zu tag steigt es aber auch richtung osten . |
| **Clean output** | später sind die schauer und gewitter dann meist nur noch im westen unterwegs aber besserung in der osthälfte und im osten sind schauer und gewitter unterwegs die von westen her seltener werden . |
| **Misaligned output** | von gewitterböen abgesehen wird es morgen vormittag nur im westen am freundlichsten aber auch in der osthälfte muss mit schauern und gewittern gerechnet werden die im westen morgen vormittag heftig sein können an der ostsee wird es sehr windig . |
| **Failure type** | acceptable |

#### Sample: `test/06May_2011_Friday_tagesschau-6435`

| | Text |
|---|---|
| **Reference** | heute nacht zwölf grad in der kölner bucht und ein grad im bayerischen wald . |
| **Clean output** | heute nacht zwölf grad am niederrhein und ein grad im bayerischen wald . |
| **Misaligned output** | am tag zwölf grad am niederrhein und ein grad im bayerischen wald werte zwischen zwanzig und achtundzwanzig grad . |
| **Failure type** | acceptable |


### Head=40%, Tail=10%

#### Sample: `test/13July_2009_Monday_tagesschau-7509`

| | Text |
|---|---|
| **Reference** | an der luftmassengrenze kommt es zu teilweise starken und gewittrigen regenfällen . |
| **Clean output** | im bergland sind zum teil heftige gewitter unterwegs . |
| **Misaligned output** | hier und da gibt es zum teil heftige gewitter und kräftige regenschauer . |
| **Failure type** | acceptable |

#### Sample: `test/03July_2009_Friday_tagesschau-2014`

| | Text |
|---|---|
| **Reference** | am sonntag vor allem im osten und süden noch schauer oder gewitter sonst überwiegend freundlich . |
| **Clean output** | am sonntag vor allem im osten und süden teilweise kräftige schauer und gewitter sonst zum teil freundlich . |
| **Misaligned output** | im osten und süden noch einzelne schauer und gewitter sonst wird es in der zweiten nachthälfte freundlicher als in der ersten nachthälfte . |
| **Failure type** | acceptable |

#### Sample: `test/06October_2011_Thursday_heute-5534`

| | Text |
|---|---|
| **Reference** | und das mit tief ophelia das die kaltluft mit kräftigen schauern richtung mitteleuropa treibt . |
| **Clean output** | die ausläufer eines tiefs bei island überqueren in der nacht den norden mitteleuropas . |
| **Misaligned output** | die kaltfront eines tiefs über dem balkan überquert bis morgen abend die mitte europas mit regenwolken ostwärts . |
| **Failure type** | acceptable |

#### Sample: `test/06September_2010_Monday_tagesschau-1153`

| | Text |
|---|---|
| **Reference** | am donnerstag noch unbeständig aber ab freitag wird es von westen langsam freundlicher . |
| **Clean output** | am donnerstag wechselhaft aber am freitag wird es von westen zunehmend freundlicher . |
| **Misaligned output** | dann wird es von westen später freundlicher . |
| **Failure type** | acceptable |

#### Sample: `test/04December_2011_Sunday_tagesschau-7791`

| | Text |
|---|---|
| **Reference** | sehr windiges wetter am dienstag regen schnee und graupelschauer . |
| **Clean output** | am dienstag regen schnee und graupelschauer . |
| **Misaligned output** | am dienstag regen schnee und graupelschauer . |
| **Failure type** | acceptable |


## Head Contam + Tail Trunc


### Head=20%, Tail=20%

#### Sample: `test/01April_2010_Thursday_tagesschau-4330`

| | Text |
|---|---|
| **Reference** | am freundlichsten ist es noch im nordosten sowie in teilen bayerns . |
| **Clean output** | am freundlichsten wird es im nordosten sowie in den mittelgebirgen . |
| **Misaligned output** | wechselhaft und am freundlichsten wird es im nordosten . |
| **Failure type** | acceptable |

#### Sample: `test/11February_2010_Thursday_tagesschau-8761`

| | Text |
|---|---|
| **Reference** | im laufe des wochenendes wird es dann immer trockener kalt bleibt es aber . |
| **Clean output** | am wochenende wird es vorübergehend trockener aber es bleibt kalt . |
| **Misaligned output** | im osten bleibt es zunächst noch freundlich aber zum wochenende wird es auch dort trockener . |
| **Failure type** | acceptable |

#### Sample: `test/08October_2009_Thursday_tagesschau-5352`

| | Text |
|---|---|
| **Reference** | an der ostsee heute nacht stürmische böen aus west . |
| **Clean output** | an der ostsee heute nacht starke bis stürmische böen . |
| **Misaligned output** | an der ostsee heute nacht starke bis stürmische böen . |
| **Failure type** | acceptable |

#### Sample: `test/14September_2010_Tuesday_heute-8239`

| | Text |
|---|---|
| **Reference** | und es kommt ein neues tief das uns morgen abend dann erreichen wird an der nordsee zum teil mit schweren sturmböen . |
| **Clean output** | ein neues tief sorgt bei uns in der nordhälfte für teilweise schwere sturmböen vereinzelt mit orkanartigen böen . |
| **Misaligned output** | die ausläufer eines nordseetiefs bestimmen in den nächsten tagen das wetter in der nordhälfte deutschlands . |
| **Failure type** | hallucination |

#### Sample: `test/22July_2009_Wednesday_tagesschau-2873`

| | Text |
|---|---|
| **Reference** | in der nacht muss vor allem in der nordwesthälfte mit schauern und gewittern gerechnet werden die heftig ausfallen können . |
| **Clean output** | in der nacht muss vor allem in der nordwesthälfte mit schauern und gewittern gerechnet werden die kräftig sein können . |
| **Misaligned output** | es sorgt verbreitet für sonnenschein und in der nacht muss vor allem in der nordwesthälfte mit kräftigen schauern und gewittern gerechnet werden die kräftig ausfallen können . |
| **Failure type** | acceptable |


### Head=10%, Tail=40%

#### Sample: `test/31March_2010_Wednesday_tagesschau-999`

| | Text |
|---|---|
| **Reference** | und nun die wettervorhersage für morgen donnerstag den ersten april . |
| **Clean output** | und nun die wettervorhersage für morgen donnerstag den ersten april . |
| **Misaligned output** | und nun die wettervorhersage für morgen mittwoch den vierten juli . |
| **Failure type** | acceptable |

#### Sample: `test/14December_2010_Tuesday_heute-650`

| | Text |
|---|---|
| **Reference** | dort auch bei leichten plusgraden aber sonst bleibt es frostig . |
| **Clean output** | allerdings gibt es schon plusgrade sonst bleibt es überall frostig . |
| **Misaligned output** | aber das ist ja schon das nächste plusgrade in den nächsten tagen . |
| **Failure type** | hallucination |

#### Sample: `test/15March_2011_Tuesday_tagesschau-3326`

| | Text |
|---|---|
| **Reference** | aber im laufe der nacht wird es zunehmend wolkiger . |
| **Clean output** | aber die nacht wird schon noch ein bisschen wolkiger . |
| **Misaligned output** | aber die nacht wird noch ein bisschen heißer . |
| **Failure type** | acceptable |

#### Sample: `test/24October_2009_Saturday_tagesschau-4286`

| | Text |
|---|---|
| **Reference** | der wind aus süd bis west weht schwach bis mäßig . |
| **Clean output** | der wind weht schwach an der see auch mäßig . |
| **Misaligned output** | dabei ist es windig . |
| **Failure type** | under-generation |

#### Sample: `test/06November_2010_Saturday_tagesschau-1162`

| | Text |
|---|---|
| **Reference** | schwacher bis mäßiger auf den bergen in böen auch starker wind in der nordhälfte aus ost bis nordost sonst aus unterschiedlichen richtungen . |
| **Clean output** | schwacher bis mäßiger im bergland zum teil frischer an der nordsee zum teil starker bis stürmischer wind aus süd bis südwest . |
| **Misaligned output** | der wind weht schwach bis mäßig im bergland auch frisch . |
| **Failure type** | under-generation |


### Head=40%, Tail=10%

#### Sample: `test/05May_2010_Wednesday_tagesschau-3360`

| | Text |
|---|---|
| **Reference** | am wochenende wird es dann etwas wärmer das ganze bei wechselhaftem wetter mit schauern gewittern und gelegentlichem sonnenschein . |
| **Clean output** | im laufe des tages wird es immer wärmer dabei unbeständiger es gibt gewitter aber auch immer wieder sonnenschein . |
| **Misaligned output** | in der nordhälfte regnet es gebietsweise richtung südosten länger anhaltend dabei gibt es wechselhaftes wetter mit gewittern und viel sonne . |
| **Failure type** | acceptable |

#### Sample: `test/08February_2010_Monday_heute-1497`

| | Text |
|---|---|
| **Reference** | es bleibt meist trüb wie auf diesem bild hier . |
| **Clean output** | und diese wolken liegen ja schon über dem bergland . |
| **Misaligned output** | die gibt es zur zeit noch im süden aber sie bildet sich nur vereinzelt über dem bergland . |
| **Failure type** | hallucination |

#### Sample: `test/17May_2010_Monday_heute-5576`

| | Text |
|---|---|
| **Reference** | auch noch verhältnismäßig kühl aber dann sieht es schon nach ein paar sonnigeren und auch deutlich wärmeren tagen aus . |
| **Clean output** | und es wird auch entsprechend kühl aber dann kann sich zumindest in den nächsten tagen erstmal besser die sonne zeigen und es wird auch wieder wärmer . |
| **Misaligned output** | in alpennähe gibt es anfangs auch noch regen und es bleibt kühl aber dann wird es langsam wieder etwas sonniger die sonne kommt raus . |
| **Failure type** | acceptable |

#### Sample: `test/05September_2009_Saturday_tagesschau-4085`

| | Text |
|---|---|
| **Reference** | in der nacht an der see noch stürmische böen . |
| **Clean output** | in der nacht stürmisch auf den bergen sturmböen . |
| **Misaligned output** | in der nacht stürmisch auf den bergen sturmböen . |
| **Failure type** | acceptable |

#### Sample: `test/24January_2013_Thursday_tagesschau-6322`

| | Text |
|---|---|
| **Reference** | am sonntag überquert und dann ein atlantischer tiefausläufer dabei wird es milder . |
| **Clean output** | am sonntag folgt dann aber ein atlantiktief und von westen wird es langsam wieder milder . |
| **Misaligned output** | die sonne lässt sich nur selten blicken aber am sonntag macht sich dann ein atlantischer atlantiktief mit etwas mildere luft bemerkbar . |
| **Failure type** | acceptable |


## Head Contam + Tail Contam


### Head=20%, Tail=20%

#### Sample: `test/08October_2009_Thursday_tagesschau-5358`

| | Text |
|---|---|
| **Reference** | im osten aber erst noch freundliche abschnitte . |
| **Clean output** | im osten bleibt es noch ein bisschen trüb . |
| **Misaligned output** | im osten bleibt es noch ein bisschen freundlich . |
| **Failure type** | acceptable |

#### Sample: `test/14July_2009_Tuesday_tagesschau-2430`

| | Text |
|---|---|
| **Reference** | heute nacht schwül warme dreizehn bis achtzehn grad . |
| **Clean output** | heute nacht dreizehn bis achtzehn grad bei auflandigem wind schwacher bis mäßiger wind . |
| **Misaligned output** | in gewitternähe heute nacht dreizehn bis achtzehn in der kölner bucht milde achtzehn grad . |
| **Failure type** | acceptable |

#### Sample: `test/21November_2011_Monday_heute-5437`

| | Text |
|---|---|
| **Reference** | ihnen noch einen schönen abend . |
| **Clean output** | und jetzt wünsche ich ihnen noch einen schönen abend . |
| **Misaligned output** | und jetzt wünsche ich ihnen noch einen schönen abend . |
| **Failure type** | acceptable |

#### Sample: `test/08September_2010_Wednesday_tagesschau-5246`

| | Text |
|---|---|
| **Reference** | am samstag regnet es noch etwas in küstennähe sonst meist freundlich und trocken . |
| **Clean output** | am samstag an den küsten noch etwas regen sonst meist freundlich und trocken . |
| **Misaligned output** | am samstag regnet es an den küsten noch zeitweise sonst ist es meist freundlich und trocken . |
| **Failure type** | acceptable |

#### Sample: `test/20April_2011_Wednesday_tagesschau-3086`

| | Text |
|---|---|
| **Reference** | auch am tag wieder viel sonnenschein später bilden sich hier und da ein paar quellwolken . |
| **Clean output** | auch am tag viel sonne im laufe des tages bilden sich später nur hier und da ein paar quellwolken . |
| **Misaligned output** | auch in der nacht am mittwoch viel sonne später ist es nur noch hier und da wolkenverhangen . |
| **Failure type** | acceptable |


### Head=10%, Tail=40%

#### Sample: `test/08April_2010_Thursday_tagesschau-3957`

| | Text |
|---|---|
| **Reference** | morgen reichen die temperaturen von neun grad an der ostsee bis siebzehn grad im breisgau . |
| **Clean output** | morgen temperaturen von neun grad an der ostsee bis siebzehn grad im breisgau . |
| **Misaligned output** | morgen temperaturen von neun grad an der ostsee bis siebzehn grad im breisgau anfangs wechselhaft und noch kühler . |
| **Failure type** | acceptable |

#### Sample: `test/31March_2010_Wednesday_tagesschau-1008`

| | Text |
|---|---|
| **Reference** | am freitag mal sonne mal wolken und nur einzelne schauer stellenweise zeigt sich die sonne auch für längere zeit . |
| **Clean output** | am freitag mal sonne mal wolken mal schauer hier und da zeigt sich die sonne auch für längere zeit . |
| **Misaligned output** | am freitag mal sonne mal wolken mal schauer hier und da zeigt sich die sonne auch für längere zeit sobald der nebel weg ist . |
| **Failure type** | acceptable |

#### Sample: `test/29April_2010_Thursday_heute-8626`

| | Text |
|---|---|
| **Reference** | und das wetter bringt dort auch noch sonnenschein am vormittag zumindest . |
| **Clean output** | und da scheint auch noch mal die sonne am vormittag . |
| **Misaligned output** | im bergland scheint dann auch immer noch die sonne am vormittag ein paar schauer am nachmittag gibt es dann im laufe des nachmittags auch dort schauer . |
| **Failure type** | acceptable |

#### Sample: `test/22September_2010_Wednesday_heute-2624`

| | Text |
|---|---|
| **Reference** | am freitag geht es aber langsam abwärts im osten gewitterschauer . |
| **Clean output** | am freitag dann langsame abkühlung und die temperaturen sinken im osten und südosten noch ein wenig . |
| **Misaligned output** | am freitag dann langsames tauwetter und steigende temperaturen im osten im osten schneit es noch ein wenig und am samstag wird es deutlich kühler . |
| **Failure type** | acceptable |

#### Sample: `test/28May_2010_Friday_tagesschau-7501`

| | Text |
|---|---|
| **Reference** | heute nacht elf grad in niederbayern und bis ein grad in den nördlichen mittelgebirgen . |
| **Clean output** | heute nacht elf grad in südostbayern und ein grad im nordwesten . |
| **Misaligned output** | heute nacht elf grad in südostbayern und ein grad im nordwesten deutschlands . |
| **Failure type** | acceptable |


### Head=40%, Tail=10%

#### Sample: `test/11November_2010_Thursday_tagesschau-3564`

| | Text |
|---|---|
| **Reference** | teilweise kräftiger regen zieht heute nacht über uns ostwärts . |
| **Clean output** | die teilweise kräftigen regenfälle breiten sich heute nacht von westen aus . |
| **Misaligned output** | am mittwoch nach mitternacht zieht der teilweise kräftige und ergiebige regen nach osten . |
| **Failure type** | acceptable |

#### Sample: `test/14July_2009_Tuesday_tagesschau-2433`

| | Text |
|---|---|
| **Reference** | besonders im südlichen bergland einzelne wärmegewitter . |
| **Clean output** | im übrigen land entwickeln sich vor allem im süden deutschlands in alpennähe einzelne wärmegewitter . |
| **Misaligned output** | schwül warme luft bestimmt unser wetter in den nächsten tagen und vor allem im süden deutschlands auch das wetter mit wärmere luft und gewittern . |
| **Failure type** | hallucination |

#### Sample: `test/20June_2011_Monday_heute-6511`

| | Text |
|---|---|
| **Reference** | und damit schönen guten abend . |
| **Clean output** | das war es für heute schönen abend noch . |
| **Misaligned output** | das meine damen und herren war es für heute schönen abend noch . |
| **Failure type** | acceptable |

#### Sample: `test/06October_2011_Thursday_heute-5535`

| | Text |
|---|---|
| **Reference** | in den alpen wird es dann sogar schneien . |
| **Clean output** | an den alpen regnet es schon kräftig . |
| **Misaligned output** | das tief über dem östlichen mitteleuropa sorgt an den alpen für ergiebige regenfälle . |
| **Failure type** | acceptable |

#### Sample: `test/24April_2010_Saturday_tagesschau-3971`

| | Text |
|---|---|
| **Reference** | dort sowie in ungünstigen muldenlagen ist bodenfrost möglich . |
| **Clean output** | dort sowie in den südlichen mittelgebirgen ist bodenfrost möglich . |
| **Misaligned output** | es sind ein paar dichtere wolken unterwegs die hier und da etwas schnee bringen und örtlich bodenfrost bringen . |
| **Failure type** | hallucination |
