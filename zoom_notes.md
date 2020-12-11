# Zoom Meeting Notes from Friday 11th December

## Purpose

Biggest hands on data challenge hackathon during CDT program.

Need partners with hackathon due to expertise.


## Partners

Nicolas Longepe
 - Use our data from Copernicus (Sentinal 1, Sentinal 2)
 - Super resolution etc.

Marc Paganini (ESA)
 - Lots of knowledge over many years for AI4EO biodiversity.
 - European Commision biodiversity work (Brexit LOL) apparently we're not banned.
 - Bringing digital technologies into the solution (not just AI)
 - Digital twins project mainly focussed on AI. Biodiversity will likely be a component.
 - Designing a new agenda with the commision
 

Cooper Elsworth (Descartes Labs)
 - Start up in US focused on remote sensor data platform (data fusion, computer vision etc.)
 - Use different remote data sets, bring them together.
 - Big project on deforrestaion. With companies, trying to check their supply changes.
   - Old growth trees worse?
   - Where is it relevant.
 
Luca (WCMC UNEP)
 - Main focus is biodiversity, protecting it around the globe.
 - May have important projects that can be linked to this challenge.
 - Output may be useful for them.

Tom Dowling (WCMC UNEP)
 - Just joined WCMC UNEP, to integrate remote sensors into workflows and advance them.
 - We need AI4EO. Innovative products, and fusion products.
 - Build tools for our organistion.
   - Flexibility.
   - Interoperability.
 - Write tools in `Python3`, beware of overtraining.
 
Andrew Fleming (BAS)
 - Centre of proposal 6 months ago.
 - Providing lectures.
 - Lots of jargon from AI4EO.
 - Training material, help with physical resources, over and above SE, SO etc.

Anita Faul (BAS)
 - Make sure that everyone has a chance to speak. Include notes in chat etc..
 
## Which data sets would be useful for this project?

Nicolas L
 - Copernicus program; consistent dataset.
 - Sentinal 1 day and night radar. Same configurations, nice time series analysis. > 10m.
 - Sentinal 2 camera, and programatic acquistion etc. Monitor length, free on multiple products. 10m.
 - Very easy to access, with processing also available online.
 
Cooper E
 - Lots of data online listed on their website. Mainly use sentinal 1 as it can see through clouds etc.
 - GEDI - Lidar based data from ISS.

Kyle Story
 - Sentinal 1/ Sentinal 2 (2015)
 - Landsats allow longer time history than that.
 - Contextual datasets, with different ammounts of time.
 - Different layers of datasets etc.
 
Marc Paganini
 - Sentinal 2 -- 20km from coastline. 
 - Europe has ambition to look at protecting 20% of coastline.
 - High res sattelites. Limited budget. Look at small features, maybe for downscaling Sentinal?
 - Few global scale projects.
 - Copernicus services. Pan-European level. 
 - Land monitoring services: 
   - Global, 
   - Pan-European,
   - Local.
 - Exposure datasets - various population datasets might be helpful.

Steven Briggs
 - Data can be anywhere (no restrictions)
 
Tom Dowling 
 - Look at thermal data from Sentinal 3 etc. ENVI. Draw diversity measures from that.
 - All parts of the EM spectrum could be useful, and the more novel could be the most productive.
 - Have to make all information actionable. Monitor for deforestation. Global Forest Watch. How does data flow?

## What big problems are there in AI4EO

TD 
 - Fragmentation 
   - small areas not sustainable for organisms etc.
   - monitoring would be v. useful.

SB 
 - Fractral structure critically important for biodiversity. 

Luca
 - Project about vegation index about fragmentation.
 - Predicting impact of the development of urban areas.
 - Use lots of old non-AI techniques.
 - Species distributions etc.
 - Improve the distribution of species measurements?

SB 
 - Landcover and topography might be useful.
 - Surface temperature, Koppen climate zone.
 - Driving factors almost endless, therefore AI is a good way forward.
 - Extreme multivariate, very context dependent.

TD 
 - Validation will be v. important especiallly at the end.

## Arduin Findeis: Who would be interested in this kind of data?

TD 
 - National governments. E.g. Chernobyl.
 - Drying out of wetland forest --> worse or better?
 - Analysis of the impact of a project on biodiversity.

SB 
 - Required by international law (e.g. Paris agreement)
 - SDGs -- development goals.
 - Environmental analysis.
 
## SM: Which tools are in use, and what questions can already be answered by them?

MP 
 - Different tools are available in the community
 - U. of Wolfsburg have very nice book. In R?
 - Martin Wegman - Biodiversity tools using species modelling using EO.
 - New set of ambitions come in this year.
 - Vulnerability of the communities to climate change.
 - Decade will involve a lot of new restoration efforts.
 - Look at drivers of changes and classify them. [5]
    - Climate Change.
    - ?
    - Alien Species.
    - Exploitation of resources.
 - Nature based solutions. Greening of cities.
 - Ecosystem accounting - new standard next year. UN, then Eurostat.
 - Ecosystem monitoring and services.

SB 
 - Ecosystem migrations, e.g. butterflys etc.

MP 
 - Phenology v. important.

SB
 - Please talk about EBVs.

MP 
 - EBVs: essential biodiversity variables.
 - Which ones are essential? Currently unclear in some ways. Lead at Geo-BON.
 
 <https://geobon.org/essential-biodiversity-variables-2020-initiative-ebv2020-for-post-2020-global-biodiversity-conservation/>
 
 - Stucture and function, and community compesition.
 - Species modelling etc. using EO.
 - EBV engines.

SB 
 - look up these references.
 - Read around the subject.
 
KS
 - Background at Descartes at scale.
 - Automated analyses which run at scale.
 - Trying to build models that can characterise useful indicators for this system.

SB
 - Point information to larger scale; reasonable extrapolation.

KS 
 - Which indicators are useful?
 - Relating to biodiversity indicators, need to relate to a lot of good ground truth for hyperparameter tuning.
 - Models useless without quantification of fit.

MS 
 - We want to make sure that it is transferable to other domains and data.
 - Training data needed. E.g. example:
   - mud flats, marshes, etc. 
   - Spent alot of time looking at small features.
   - Took a lot of time to get the training and clean it well.
   - Takes a large ammount of time.
   - Maybe we should focus on places where this is available.

SB
 - 3 months extent.
 - Areas might be extended into a PhD project etc. to allow new data.


TD 
 - Lots of good data availble. Varying qualities of data. 

SB
 - Available sattelite types.
 - Information on needs from UNEP.

## Herbie Bradley, what is there that AI could solve, or has been shown to be able to solve?

MP 
 - AI neccesary because of the diversity of the disturbances meaning that a hand tuned model would be impossible.
 - Old variety of disturbance cannot be solved with traditional modelling, no way.
 - Lots of discussions about what type of classification systems to use.
 - Land cover mapping not what biodiversity community needs. They need habitat classifications, but this is very difficult, as the classes are very similar.
 - Hopefully it is better than traditional.
 
SB 
 - Bringing together different sorts of data could leads to new innovation.
 - Impossible with a standard physical model, so AI provides something new.
 - No such as a single land cover classification.
 - 1st ever LC algorithm. 1 cover called arable, but a lot of different crops. 
 - 10 different croplands, only one natural grassland.
 - classification is dependent on the purpose of the algorithm (e.g. farming).
 - Different consequences from the biodiversity of the system to each particular habitat.

NL
 - Fuse AI and bio-observation.
 - Try to look at fragmentation? Would need high resolution, but that is too expensive at scale.
 - 2 ways to do AI:
   - to fuse AI and EO
   - to combine biodiversity indicators.
 - Lots of different measures for fragmentation.
 - Many species require grassland adjacent to woodlands (e.g. )
 
MP 
 - Some open source tools available.

## SB: Methodology of how the project operates. Group activity etc.

Scott Hosking:
 - Useful to know different strengths etc.
 - Be flexible about how you work.
 - Reproduce something that is out there.
 - Low hanging fruit.
 - Engage with Anita, S Briggs, Scott, the partners.
 - Train yourself to go off and do independent research for next year.
 - Lots of work - need to focus tightly on something feasible and useful.
 - Get in touch with Partners.
 
SB 
 - Get in touch with students.
 
CE 
 - Python API availalbe for students (contact to be given a key).
 
TD 
 - Lots of help available

AF 
 - `Jasmin` facility data available 
 
SB 
 - Building links is incredibly valuable and this is a brilliant opportunity for us all. 
 - Normally ESA takes ages to do this, so get stuck in.
 
## What are the main challenges with the datasets.

MP
 - Takes a long time to agree with EBVs.
 - Therefore moving into production is difficult.
 - Rely a lot on peer-review papers to work out what EBVs are.
 - Community agreement useful. ESA strong engagement on developming EBVs by Copernicus.
 - Extremely complex problem.
 - Datasets not yet available in general. Some are. GEOBON creating EBV engines on cloud with AZURE and EBV portal.

SB
 - Relative imaturity of definition of EBVs.
 - Substantial contributions possible.
 - Enough work to get started.
 
MP 
 - Links difficult to find, will send to Anita later.

From Simon Mathis to Everyone: (5:08 pm)
 - Yes, a pointer to those resources would be extremely helpful (:
 
From Tom Dowling (UNEP-WCMC) to Everyone: (5:11 pm)
 - Probably too technical/getting towards EO for EO problem, but re scale and landcover. Exactly as the others say the detail (scale) of the information available is always a limitation. So one of the cutting edge fields is sub-pixel information extraction. The old school method is pan-sharpening, but people are looking at method of AI to achieve the same for a wide variety of sensors for a variety of needs.
 
From Tom Dowling (UNEP-WCMC) to Everyone: (5:15 pm)
We have an account on JASMIN-CEDA as well just fyi if that helps/it is a space we work in as well.

From Nicolas LONGEPE  to Everyone: (5:13 pm)
- about AI and SR and vegetation, you maylook at this link 
- <https://kelvins.esa.int/proba-v-super-resolution/>

TD
 - Ecosystem structure and functioning most useful.

SB 
 - Natural capital accounting.
 
From Marc Paganini to Everyone: (5:25 pm)
- Intergovernmental science-policy Platform for Biodiversity and Ecosystem Services (IPBES) (2019), Global Assessment Report on Biodiversity and Ecosystem Services. 
- <https://ipbes.net/global-assessment>
- UN SEEA Ecosystem Accounting <https://seea.un.org/ecosystem-accounting>
- European Commission, EU Biodiversity Strategy for 2030, COM(2020) 380 <https://ec.europa.eu/environment/nature/biodiversity/strategy/index_en.htm>
- Pereira H.M., Ferrier S., Walters M., Geller G.N., Jongman R.H.G., Scholes R.J. et al. (2013), Essential biodiversity variables, Science 339: 277–278. -
- <http://science.sciencemag.org/content/339/6117/277>
- From luca - UNEP WCMC to Everyone: (5:26 pm)
- <https://encore.naturalcapital.finance/en> to explore natural capital
- From Cooper Elsworth - Descartes Labs to Everyone: (5:26 pm)
- Thank you - we’re very excited to see the progress!
- From Marc Paganini to Everyone: (5:26 pm)
- Global Biodiversity Outlook (GBO) <https://www.cbd.int/gbo/>
- The Essential Biodiversity Variables (EBV) <http://geobon.org/essential-biodiversity-variables/>
- Mapping and Assessment of Ecosystems and their Services (MAES) <https://biodiversity.europa.eu/ecosystems>
- From Marc Paganini to Everyone: (5:27 pm)
- Biodiversity Information System for Europe (BISE) <https://biodiversity.europa.eu>
- BiodivERsA <https://www.biodiversa.org/>

## Zoom Messages:

Standard publicly-available datasets available through the Descartes Labs data platform: 
 - <https://www.descarteslabs.com/standard-data-products/>

Another modelling effort, at the ecosystem level that is similar to what Marc is talking about That we have used in past: 
- <https://invest.readthedocs.io/en/latest/index.html>

