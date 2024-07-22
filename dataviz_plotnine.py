
#goals -----------------------------------
#visualizations in python using ggplot2 style with plotnine package

#packages
from plotnine.data import economics_long
from plotnine import ggplot, aes, geom_line, facet_wrap, labs, theme_xkcd,scale_x_datetime
from mizani.breaks import date_breaks
from mizani.formatters import date_format

#check data
economics_long.head(4)

#plot data
(ggplot(economics_long)
+ aes(x =  "date", y = "value")
+ geom_line()
+ facet_wrap("variable", scales="free")
+ scale_x_datetime(
    breaks = date_breaks("15 years"), 
    labels = date_format("%Y"))
+ labs(
    x = "Period(years)",
    y = "",
    title = "US economic indicators")
+ theme_xkcd())

