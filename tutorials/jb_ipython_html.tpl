{%- extends 'basic.tpl' -%}

{%- block header -%}
<!DOCTYPE html>
<html>
<head>

<meta charset="utf-8" />
<title>{{resources['metadata']['name']}}</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>

{% for css in resources.inlining.css -%}
    <style type="text/css">
    {{ css }}
    </style>
{% endfor %}

<style type="text/css">
// Overrides of notebook CSS for static HTML export 
body {
  overflow: visible;
  background-color: #333;
  padding: 8px;
  margin:auto;
}

div#notebook_panel{
  color=#333 
  background-color: #4B4B4B;
}
div#notebook {
  background: #fff;
  color: #333;
  width: 120ex;
  margin:auto;
  padding-left: 1em;
  padding-right: 4em;
  padding-top: 2ex;
  overflow: visible;
  border-top: none;
}

@media print {
   h1,
  h2,
  h3,
  h4,
  h5,
  h6 {
    page-break-after: avoid; // Prevent headings from being printed at the bottom of the page
  }
 
  article {
    page-break-before: always; // Always start new articles on a new page
  }
 
  img {
    page-break-inside: avoid; // Prevent images from being split up
  }

  a:after {
     display: none !important;
  }

  div.cell {
    display: block;
    page-break-inside: avoid;
  } 

  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }

  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>


<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="styles/jb_custom_ipython.css">

<!-- Loading mathjax macro -->
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<!-- End of mathjax configuration -->

<!-- Google analytics-->
    <script>
      (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
      (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
      m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
      })(window,document,'script','//www.google-analytics.com/analytics.js','ga');
      
      ga('create', 'UA-54852702-1', 'auto');
      ga('send', 'pageview');
    </script>
<!-- End Google analytics -->

</head>
{%- endblock header -%}

{% block body %}
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">
{{ super() }}
    </div>
  </div>
</body>
{%- endblock body %}

{% block footer %}
</html>
{% endblock footer %}
