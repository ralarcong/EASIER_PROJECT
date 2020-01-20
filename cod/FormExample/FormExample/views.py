#importing required packages
from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render_to_response
from django.views.decorators.csrf import csrf_exempt
import sys
#sys.path.insert(0, '/home/users/ralarcon/Rodrigo/DOC/MLESSUI/cod')
sys.path.insert(0,'C:/Users/RODRIGO/Documents/UC3M/DOC/DOC/MLESSUI/cod')

import main
#disabling csrf (cross site request forgery)
@csrf_exempt
def index(request):
    #if post request came
    if request.method == 'POST':
        #getting values from post
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        web = request.POST.get('web')

        #adding the values in a context variable
        context = {
            'name': name,
            'email': email,
            'phone': phone,
            'web': web
        }

        #getting our showdata template
        if 'SimplificarURL' in request.POST:
            main.main(web)
            return render_to_response('temp.html')
        elif 'SimplificarTexto' in request.POST:
            main.main(web)
            return render_to_response('temp2.html')
        elif 'SimplificarURLSyn' in request.POST:
            main.main(web)
            return render_to_response('tempdictionary.html')
        elif 'SimplificarTextoSyn' in request.POST:
            main.main(web)
            return render_to_response('tempdictionary.html')
        #returing the template
        #return HttpResponse(template.render(context, request))
	
    else:
        #if post request is not true
        #returing the form template
        template = loader.get_template('index.html')
        return HttpResponse(template.render())