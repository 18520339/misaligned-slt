# Sample Translation Comparison

Showing random samples for each misalignment type at **20% severity**.


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


## Head Trunc + Tail Trunc (20%)

### Sample: `test/05February_2010_Friday_tagesschau-2924`

| | Text |
|---|---|
| **Reference** | dabei ist es teils stark bewölkt oder neblig trüb teils freundlich mit sonnenschein . |
| **Clean output** | aber sonst ist es meist stark bewölkt oder neblig trüb teils neblig trüb teils sonnig . |
| **Misaligned output** | wolkenverhangen stellenweise bildet sich nebel sonst ist es teilweise freundlich . |
| **Failure type** | acceptable |

### Sample: `test/25November_2009_Wednesday_tagesschau-7677`

| | Text |
|---|---|
| **Reference** | am freitag wechselhaftes schauerwetter . |
| **Clean output** | am freitag regenschauer und gewitter im wechsel . |
| **Misaligned output** | am freitag regenschauer und gewitter im wechsel . |
| **Failure type** | acceptable |

### Sample: `test/09August_2010_Monday_heute-5890`

| | Text |
|---|---|
| **Reference** | ganz anders die trockene hitze über russland mit fünfunddreißig bis vierzig grad . |
| **Clean output** | und dort haben wir morgen auch trockenes sonniges wetter mit fünf bis vier grad . |
| **Misaligned output** | an der küste bei russland morgen trockenes wetter mit viel sonnenschein fünf bis fünf grad . |
| **Failure type** | acceptable |

### Sample: `test/22February_2010_Monday_tagesschau-3246`

| | Text |
|---|---|
| **Reference** | im südosten ist es heute nacht trocken sonst regnet es zeitweise in der nordhälfte teilweise kräftig . |
| **Clean output** | im norden und nordosten bleibt es heute nacht meist trocken sonst regnet es zeitweise im norden und osten zum teil ergiebig . |
| **Misaligned output** | heute nacht bleibt es meist trocken aber später beginnt es im süden wieder zu regnen . |
| **Failure type** | acceptable |

### Sample: `test/26July_2010_Monday_tagesschau-6266`

| | Text |
|---|---|
| **Reference** | am mittwoch breiten sich von nordwesten neue wolken mit regen oder schauern aus örtlich sind auch gewitter dabei . |
| **Clean output** | am mittwoch breiten sich von nordwesten schauer und gewitter aus die örtlich unwetterartig sein können . |
| **Misaligned output** | von nordwesten breiten sich später regenschauer und gewitter aus die stellenweise kräftig sein können . |
| **Failure type** | acceptable |


## Head Trunc + Tail Contam (20%)

### Sample: `test/27May_2010_Thursday_tagesschau-8382`

| | Text |
|---|---|
| **Reference** | am sonntag überall unbeständiges wetter teilweise kräftige regenfälle teilweise auch gewitter . |
| **Clean output** | am sonntag unbeständiges wetter teilweise regnet es kräftig auch gewitter sind mit dabei . |
| **Misaligned output** | wechselhaft geht es dann auch in den nächsten tagen weiter teilweise mit kräftigem regen auch einzelne gewitter sind dabei . |
| **Failure type** | acceptable |

### Sample: `test/05September_2009_Saturday_tagesschau-4085`

| | Text |
|---|---|
| **Reference** | in der nacht an der see noch stürmische böen . |
| **Clean output** | in der nacht stürmisch auf den bergen sturmböen . |
| **Misaligned output** | dabei gibt es stürmische böen . |
| **Failure type** | acceptable |

### Sample: `test/14August_2009_Friday_tagesschau-75`

| | Text |
|---|---|
| **Reference** | örtlich schauer oder gewitter die heftig sein können . |
| **Clean output** | hier und da gibt es schauer oder gewitter die heftig sein können . |
| **Misaligned output** | hier und da gibt es schauer oder gewitter die heftig sein können . |
| **Failure type** | acceptable |

### Sample: `test/03February_2010_Wednesday_heute-2362`

| | Text |
|---|---|
| **Reference** | ich wünsche ihnen einen schönen abend und machen sie es gut . |
| **Clean output** | ihnen noch einen schönen abend und machen sie es gut . |
| **Misaligned output** | guten abend liebe zuschauer . |
| **Failure type** | under-generation |

### Sample: `test/16December_2009_Wednesday_tagesschau-5907`

| | Text |
|---|---|
| **Reference** | und nun die wettervorhersage für morgen donnerstag den siebzehnten dezember . |
| **Clean output** | und nun die wettervorhersage für morgen donnerstag den siebzehnten dezember . |
| **Misaligned output** | und nun die wettervorhersage für morgen donnerstag den siebzehnten dezember . |
| **Failure type** | acceptable |


## Head Contam + Tail Trunc (20%)

### Sample: `test/21May_2010_Friday_tagesschau-5094`

| | Text |
|---|---|
| **Reference** | abgesehen von schauer und gewitterböen weht der wind schwach bis mäßig aus nordwest bis nord . |
| **Clean output** | dazu weht ein schwacher bis mäßiger wind aus unterschiedlichen richtungen . |
| **Misaligned output** | bei schauern und gewittern sind starke böen möglich sonst schwacher bis mäßiger wind aus unterschiedlichen richtungen . |
| **Failure type** | acceptable |

### Sample: `test/05September_2009_Saturday_tagesschau-4085`

| | Text |
|---|---|
| **Reference** | in der nacht an der see noch stürmische böen . |
| **Clean output** | in der nacht stürmisch auf den bergen sturmböen . |
| **Misaligned output** | in der nacht starke bis stürmische böen an der see sturmböen . |
| **Failure type** | acceptable |

### Sample: `test/03February_2010_Wednesday_heute-2355`

| | Text |
|---|---|
| **Reference** | im norden wird es dagegen noch trocken sein . |
| **Clean output** | im norden bleibt es meist trocken . |
| **Misaligned output** | im norden bleibt es meist trocken . |
| **Failure type** | acceptable |

### Sample: `test/26August_2009_Wednesday_tagesschau-3213`

| | Text |
|---|---|
| **Reference** | und nun die wettervorhersage für morgen donnerstag den siebenundzwanzigsten august . |
| **Clean output** | und nun die wettervorhersage für morgen donnerstag den siebenundzwanzigsten august . |
| **Misaligned output** | und nun die wettervorhersage für morgen donnerstag den siebenundzwanzigsten juli . |
| **Failure type** | acceptable |

### Sample: `test/29August_2009_Saturday_tagesschau-5020`

| | Text |
|---|---|
| **Reference** | in den frühstunden kann sich vereinzelt nebel bilden . |
| **Clean output** | morgen früh bildet sich stellenweise nebel . |
| **Misaligned output** | wolkenlücken gibt es in den frühen morgen hier und da an den küsten stellenweise wieder nebel oder hochnebel . |
| **Failure type** | acceptable |


## Head Contam + Tail Contam (20%)

### Sample: `test/08February_2010_Monday_heute-1498`

| | Text |
|---|---|
| **Reference** | und im westen haben wir kaum noch schnee aber es gibt eben auch ausnahmen die mittelgebirge . |
| **Clean output** | in der westhälfte kaum noch schnee und die schneefallgrenze sinkt weiter in den nächsten tagen aber auf den gefrierpunkt in den mittelgebirgen . |
| **Misaligned output** | in der westhälfte kaum schneefälle die sich auflösen vor allem auf den bergen der westhälfte . |
| **Failure type** | hallucination |

### Sample: `test/02October_2010_Saturday_tagesschau-1300`

| | Text |
|---|---|
| **Reference** | sehr milde luft strömt zu uns . |
| **Clean output** | dabei strömt sehr milde luft zu uns . |
| **Misaligned output** | dabei strömt sehr milde luft zu uns nach deutschland . |
| **Failure type** | acceptable |

### Sample: `test/17May_2010_Monday_heute-5576`

| | Text |
|---|---|
| **Reference** | auch noch verhältnismäßig kühl aber dann sieht es schon nach ein paar sonnigeren und auch deutlich wärmeren tagen aus . |
| **Clean output** | und es wird auch entsprechend kühl aber dann kann sich zumindest in den nächsten tagen erstmal besser die sonne zeigen und es wird auch wieder wärmer . |
| **Misaligned output** | die nächsten tage werden auch ziemlich kühl aber dann kann sich die sonne zumindest in den nächsten tagen langsam wieder etwas häufiger zeigen und die wärme kommt auch nach deutschland . |
| **Failure type** | acceptable |

### Sample: `test/26October_2009_Monday_tagesschau-1403`

| | Text |
|---|---|
| **Reference** | am freitag scheint abseits der nebelgebiete häufig die sonne . |
| **Clean output** | am freitag nach auflösung von teilweise zähen nebelfeldern viel sonne . |
| **Misaligned output** | am freitag abseits des nebels viel sonne in der nordhälfte . |
| **Failure type** | acceptable |

### Sample: `test/23August_2010_Monday_heute-5328`

| | Text |
|---|---|
| **Reference** | ja und so bleibt es dann auch erstmal am mittwoch . |
| **Clean output** | auch am mittwoch ändert sich noch wenig an diesem wetter . |
| **Misaligned output** | und das wird auch am mittwoch noch der fall sein . |
| **Failure type** | acceptable |
