navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia ||
    (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) ?
        function(c, os, oe) {
            navigator.mediaDevices.getUserMedia(c).then(os,oe);
        } : null ||
    navigator.msGetUserMedia;

window.URL = window.URL ||
    window.webkitURL ||
    window.msURL ||
    window.mozURL;

// http://stackoverflow.com/questions/6524288
$.fn.pressEnter = function(fn)
{

    return this.each(function() {
        $(this).bind('enterPress', fn);
        $(this).keyup(function(e){
            if(e.keyCode == 13)
            {
              $(this).trigger("enterPress");
            }
        })
    });
 };

function registerHbarsHelpers() {
    // http://stackoverflow.com/questions/8853396
    Handlebars.registerHelper('ifEq', function(v1, v2, options) {
        if(v1 === v2) {
            return options.fn(this);
        }
        return options.inverse(this);
    });
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
  

//zwidth and zheight is the size of the frame for the server side to process
var zwidth = 2560; //1920; //1280; //448; //640; //800 //1280
var zheight = 1440; //1080; //720; //252; //360; // 450  // 720

var camera_width = 2560; //1920; //1280;
var camera_height = 1440 ; //1080; //720;

//The dimension of the image encoded by the server and send to the client
var ui_width = 928; //816;
var ui_height = 522; //459;

var display_image_size = 96;

var matched_result_max_show = 500;
var unmatched_result_max_show = 50000;

var sendFrameLoopImgQuality = 0.7;
var learnmsg = "None";
var statemsg = "None";
var hasstatemsg = "None";
var aug_all_existing = "None";

function startServer()
{
        var msg = {
            'type': 'START',
        };
        socket.send(JSON.stringify(msg));             
}

function sendFrameLoop()
{
    if (socket == null || socket.readyState != socket.OPEN ||
        !vidReady || numNulls != defaultNumNulls) {
        console.log("sendFrameLoop is not valid")
        return;
    }

    if (tok > 0)
    {
    //     var startTime, endTime;
    //     startTime = new Date();

    //     //var canvas = document.createElement('canvas');
    //     //canvas.width = zwidth;
    //     //canvas.height = zheight;
    //     //var cc = canvas.getContext('2d');
    //     //cc.drawImage(vid, 0, 0, zwidth, zheight);
    //     //var apx = cc.getImageData(0, 0, zwidth, zheight);

    //     // var canvas = document.getElementById('videoel');
    //     // var cc = canvas.getContext('2d');
    //     // var apx = cc.getImageData(0, 0, camera_width, camera_height);

    //     //2nd argument is image quality, from 1 (full quality) to 0 (zero quality), default 0.92
    //     //var dataURL = canvas.toDataURL('image/jpeg', sendFrameLoopImgQuality)
    //     //console.log(dataURL);
        var msg = {};
        if(hasstatemsg != "None")
        {
            msg = {'type': 'FRAME', 'learnmsg': learnmsg, 'hasstatemsg': hasstatemsg, 'statemsg': "None", 'aug_all_existing': aug_all_existing};
        }  
        else
        {
            msg = {'type': 'FRAME', 'learnmsg': learnmsg, 'hasstatemsg': hasstatemsg, 'statemsg': statemsg, 'aug_all_existing': aug_all_existing};
        }

       
        socket.send(JSON.stringify(msg));
        //console.log("sendFrameLoop send frame 0...");
        tok--;

        if(learnmsg != "None")
        {
            console.log("Just send frame loop together with picked unknown person")
            learnmsg = "None";
        }

        if(statemsg != "None" && hasstatemsg == "None")
        {
            console.log("Just send frame loop together with all state info")
            statemsg = "None";
        }   

        if(hasstatemsg != "None")
        {
            console.log("Just send frame loop with the hasstatemsg hint")
            hasstatemsg = "None";
        }           

        if(aug_all_existing != "None")
        {
            console.log("Just send frame loop together with augment all existing person")
            aug_all_existing = "None";            
        }
    }

    // if (tok2 > 0)
    // {
    // //     var startTime, endTime;
    // //     startTime = new Date();

    // //     //var canvas = document.createElement('canvas');
    // //     //canvas.width = zwidth;
    // //     //canvas.height = zheight;
    // //     //var cc = canvas.getContext('2d');
    // //     //cc.drawImage(vid, 0, 0, zwidth, zheight);
    // //     //var apx = cc.getImageData(0, 0, zwidth, zheight);

    // //     // var canvas = document.getElementById('videoel');
    // //     // var cc = canvas.getContext('2d');
    // //     // var apx = cc.getImageData(0, 0, camera_width, camera_height);

    // //     //2nd argument is image quality, from 1 (full quality) to 0 (zero quality), default 0.92
    // //     //var dataURL = canvas.toDataURL('image/jpeg', sendFrameLoopImgQuality)
    // //     //console.log(dataURL);

    //     var msg = {'type': 'FRAME', 'id': 1};
    //     socket.send(JSON.stringify(msg));
    //     console.log("sendFrameLoop send frame 1...");
    //     tok2--;
    // }
    //     tok--;

    //     endTime = new Date();
    //     var timeDiff = endTime - startTime; //in ms
    //     console.log("sendFrameLoop of cam 0 takes: " + timeDiff + " ms");
    // }

    // if (tok2 > 0)
    // {
    //     var startTime, endTime;
    //     startTime = new Date();

    //     //var canvas = document.createElement('canvas');
    //     //canvas.width = zwidth;
    //     //canvas.height = zheight;
    //     //var cc = canvas.getContext('2d');
    //     //cc.drawImage(vid2, 0, 0, zwidth, zheight);
    //     //var apx = cc.getImageData(0, 0, zwidth, zheight);

    //     // var canvas = document.getElementById('videoel');
    //     // var cc = canvas.getContext('2d');
    //     // var apx = cc.getImageData(0, 0, camera_width, camera_height);

    //     //2nd argument is image quality, from 1 (full quality) to 0 (zero quality), default 0.92
    //     //var dataURL = canvas.toDataURL('image/jpeg', sendFrameLoopImgQuality)
    //     //console.log(dataURL);

    //     var msg = {
    //         'type': 'FRAME',
    //         'id': 1,
    //         'dataURL': '',
    //         'identity': defaultPerson
    //     };
    //     socket.send(JSON.stringify(msg));
    //     tok2--;

    //     endTime = new Date();
    //     var timeDiff = endTime - startTime; //in ms
    //     console.log("sendFrameLoop of cam 1 takes: " + timeDiff + " ms");
    // }

    setTimeout(function() {requestAnimFrame(sendFrameLoop)}, 5);
}

var max_show_pic = 2;
var info = {'-1': 0};

function getPeopleInfoHtml()
{
    var startTime, endTime;
    startTime = new Date();
    console.log(people)
    console.log(identity_ofppl)
    var len = people.length;

    for (var i = 0; i < len; i++) {
        info[i] = 0;
    }

    var len = images.length;
    console.log(images)
    for (var i = 0; i < len; i++)
    {
        var id = identity_ofppl.findIndex(k => k==images[i].identity);
        info[id] += 1;
    }

    var unlearn_num =0;
    for(var pp =result_table.length-1; pp>=0; pp--)
    {
        if (result_table[pp].IsSelected == false)
        {    
            unlearn_num += 1;
        }
    }

    var h = "<li><b>Total person:</b> "+ people.length +"</li>";
    h += "<li><b>Total images:</b> "+ images.length +"</li>";
    h += "<li><b>Total unlearn ppl:</b> "+ unlearn_num +"</li>";
    var len = people.length;

    //List from the latest to oldest
    for (var i = len-1; i >=0 ; i--)
    {
        h += "<li style='color:red;font-size:18px'><b><a href='javascript:openPersonImages(" + identity_ofppl[i] + ")'> " +people[i]+"</a>:</b> "+info[i];
        if (info[i]>0)
        {
            h += "<a href='javascript:deletePerson(" + identity_ofppl[i] + ")'>(X)</a>";
        }

        h += "<input type='checkbox' id='m" + identity_ofppl[i] + "' value='" + identity_ofppl[i] + "'>";
        h += "</li>";

        //Show first few pics of the known person
        if (info[i]>0)
        {
            h += "<br>";
            var pic_to_show = Math.min(max_show_pic, info[i]);
            for(var kz=0; kz<pic_to_show; kz++)
            {
                var imgIdx = findImageByIdentity(identity_ofppl[i], kz);
                h += "<img src=" + images[imgIdx].image + " width='72' height='72' title='Identity index: " + identity_ofppl[i] + "'>";
            }
            h += "<br><br>";
        }
    }
    endTime = new Date();
    var timeDiff = endTime - startTime; //in ms
    console.log("getPeopleInfoHtml() takes: " + timeDiff + " ms");
    return h;
}

//Redraw the list of added person at the bottom of the page based on var people and images
function redrawPeople()
{
    var startTime, endTime;
    startTime = new Date();
    //var context = {people: people, images: images, identity_ofppl: identity_ofppl};
    //$("#peopleTable").html(peopleTableTmpl(context));

    var context = {people: people, identity_ofppl: identity_ofppl};
    $("#defaultPersonDropdown").html(defaultPersonTmpl(context));

    $("#peopleInfo").html(getPeopleInfoHtml());
    endTime = new Date();
    var timeDiff = endTime - startTime; //in ms
    console.log("redrawPeople() takes: " + timeDiff + " ms");
}

function redrawResultTable()
{
    var startTime, endTime;
    startTime = new Date();

    var hu ="<table border='1'>";
    var hd ="<table border='1'>";
    var hu_counter =0;
    var hd_counter =0;
    for(var p =result_table.length-1; p>=0; p--)
    {
        if(result_table[p].bestmatchidentity>=0 && hu_counter <= matched_result_max_show)
        {
            hu += "<tr><td>";
            if( result_table[p].IsSelected==false)
            {
                hu += ( "<img src=" + result_table[p].displayimage + " width='96' height='96' onclick='AddCorrectMatch(" + p + ")'>");
            }
            else
            {
                hu += ( "<img src=" + result_table[p].displayimage + " width='96' height='96'>");
                hu += "<br><font color='red'>Added</font>";
            }
            
            hu += "</td><td>";
            //var imgIdx = findImageByIdentity(result_table[p].bestmatchidentity);
            //hu += ( "<img src=" + images[imgIdx].image + ">");
            //hu += ( "<img src=" + result_table[p].recordimage + " width='48' height='48'>");

            if (result_table[p].recordimageindex == -1)
            {
                var index = findImageByIdentity(result_table[p].bestmatchidentity);
                if (index == -1)
                {
                    result_table[p].recordimageindex == -99;
                }
                else result_table[p].recordimageindex = index;

            }

            if (result_table[p].recordimageindex >= 0)
            {
                if( result_table[p].IsSelected==false)
                {
                    hu += ( "<img src=" + images[result_table[p].recordimageindex].image + " width='96' height='96' onclick='AddCorrectMatch(" + p + ")'>");
                }
                else
                {
                    hu += ( "<img src=" + images[result_table[p].recordimageindex].image + " width='96' height='96'>");                    
                }
            }
            else
            {
                hu += "No image";
            }


            hu += "</td></tr><tr>";
            hu += "<tr><td colspan='2'>"
            hu += (result_table[p].time);

            //var idx = identity_ofppl.findIndex(k => k==result_table[p].bestmatchidentity)
            //hu += ("<br>Name: " +  people[idx]);
            hu += ("<br>Name: " +  result_table[p].targetname);
            hu += (" [Score:" +  result_table[p].simscore + "]");
            hu += "<br></td><tr>";

            hu_counter ++;
        }
        else if (hd_counter <= unmatched_result_max_show)
        {
            hd += "<tr><td>";
            hd += ( "<img src=" + result_table[p].displayimage + " width='96' height='96'>");
            hd += "</td><td>";
            hd += "Unknown<br>(id:" + p + ")";
            //hd += "<input type='radio' id='result_table_" + p + "' value='" + p + "'>"
            hd += "</td></tr><tr>";
            hd += "<tr><td colspan='2'>"
            hd += (result_table[p].time);
            hd += "<br></td><tr>";

            hd_counter ++;
        }

        // if(result_table[p].bestmatchidentity>=0)
        // {
        //     h += "<font color=green>Matched</font><br>"
        // }
        // else
        // {
        //     h += "<font color=red>Not matched!</font><br>"
        // }

        // if(result_table[p].bestmatchidentity>=0)
        // {
        //     hu += ("<br>Name: " +  people[result_table[p].bestmatchidentity]);
        //     hu += (" [Score:" +  result_table[p].simscore + "]");
        // }
        // hu += "<br></td><tr>";
    }
    hu += "</table>";
    hd += "</table>";
    $("#result_table_up").html(hu);
    $("#result_table_down").html(hd);
    $("#matched_result").html("<h4><font color=green>Matched results (" + hu_counter + ")</font></h4>");
    $("#unmatched_result").html("<h4><font color=red>Unmatched results (" + hd_counter + ")</font></h4>");

    endTime = new Date();
    var timeDiff = endTime - startTime; //in ms
    console.log("redrawResultTable() takes: " + timeDiff + " ms");
}

function AddCorrectMatch(res_idx)
{
    var r = confirm("Confirm this match is correct and add to the database");
    if (r == true) {
        var img_list_dataurl = [];
        img_list_dataurl.push(result_table[res_idx].displayimage);
        learnmsg = {
            'type': 'LEARN_UNKNOWN_PERSON',
            'img_list_dataurl': img_list_dataurl,
            'name': result_table[res_idx].targetname,
            'IsAug': false,
            'MergeExisting': result_table[res_idx].bestmatchidentity};
        console.log("Added correct match to database...");        
        result_table[res_idx].IsSelected = true;
        redrawResultTable();
    } 
}

function getDataURLFromRGB(rgb)
{
    var startTime, endTime;
    startTime = new Date();

    var rgbLen = rgb.length;
    console.log("rgb.length: " + rgbLen);

    var canvas = $('<canvas/>').width(display_image_size).height(display_image_size)[0];
    var ctx = canvas.getContext("2d");
    var imageData = ctx.createImageData(display_image_size, display_image_size);
    var data = imageData.data;
    var dLen = data.length;
    var i = 0, t = 0;

    for (; i < dLen; i +=4) {
        data[i] = rgb[t+2];
        data[i+1] = rgb[t+1];
        data[i+2] = rgb[t];
        data[i+3] = 255;
        t += 3;
    }
    ctx.putImageData(imageData, 0, 0);

    var ppcanvas = document.createElement("CANVAS");
    ppcanvas.width = display_image_size;
    ppcanvas.height = display_image_size;
    var ppctx = ppcanvas.getContext('2d');

    var zzimgData = ctx.getImageData(0, 0, display_image_size, display_image_size);
    ppctx.putImageData(zzimgData, 0, 0);

    endTime = new Date();
    var timeDiff = endTime - startTime; //in ms
    console.log("getDataURLFromRGB() takes: " + timeDiff + " ms");

    return ppcanvas.toDataURL("image/jpeg");
    //return canvas.toDataURL("image/png");
}

function updateRTT_null() {
    var diffs = [];
    for (var i = 5; i < defaultNumNulls; i++) {
        diffs.push(receivedTimes[i] - sentTimes[i]);
    }
    $("#rtt-"+socketName+"-null").html(
        jStat.mean(diffs).toFixed(2) + " ms (σ = " +
            jStat.stdev(diffs).toFixed(2) + ")"
    );
}

function updateRTT_frame() {
    var diffs = [];
    for (var i = defaultNumNulls+1; i < 2*defaultNumNulls-1; i++) {
        diffs.push(receivedTimes[i] - sentTimes[i]);
    }
    $("#rtt-"+socketName+"-frame").html(
        jStat.mean(diffs).toFixed(2) + " ms (σ = " +
            jStat.stdev(diffs).toFixed(2) + ")"
    );
}

//Only call once when numNulls == defaultNumNulls
function sendState()
{
    statemsg = {
        'type': 'ALL_STATE',
        'images': images,
        'people': people,
        'training': training,
        'people_ide': identity_ofppl
    };
    hasstatemsg = "Yes";
    //socket.send(JSON.stringify(msg));
    console.log("Send out state msg");
}

function Refresh_ALL()
{
    redrawPeople();
    redrawResultTable();
}

var counter=0;
function createSocket(address, name) // called by index.html
{
    socket = new WebSocket(address);
    socketName = name;
    socket.binaryType = "arraybuffer";

    socket.onopen = function() {
        console.log("websocket is opened")
        $("#serverStatus").html("Connected to " + name);
        sentTimes = [];
        receivedTimes = [];
        sendtime = new Date();
        receivetime = new Date();
        timediff = 0.01;
        tok = defaultTok;
        numNulls = 0;
        ccount = 0;
        ncount = 0;
        IsPrevHasBBs = [0,0,0,0,0,0,0,0,0,0,0];

        socket.send(JSON.stringify({'type': 'NULL'}));
        sentTimes.push(new Date());
    }

    socket.onmessage = function(e)
    {
        console.log(e);
        j = JSON.parse(e.data)
        if (j.type == "NULL") // sent by server for the initial RTT estimation
        {
            receivedTimes.push(new Date());
            numNulls++;
            if (numNulls == defaultNumNulls)
            {
                console.log("numNulls == defaultNumNulls")
                updateRTT_null();
                sendState();
                startServer();
                sendFrameLoop();
            } else {
                console.log("send NULL for RTT")
                socket.send(JSON.stringify({'type': 'NULL'}));
            }
            sentTimes.push(new Date());
        }
        else if (j.type == "PROCESSED")
        {
            // receivedTimes.push(new Date());
            // sentTimes.push(new Date());
            // tok++;
            // counter++;
            // if(counter == defaultNumNulls)
            // {
            //     updateRTT_frame();
            // }
        }
        else if (j.type == "NEW_PERSON") // sent by server for automatic background learning
        {
            console.log("receive new person...");
            people.push(j.val);
            identity_ofppl.push(j.identity);
            defaultPerson = j.identity
            redrawPeople();
            changeDefaultPersonCallback();
        }
        else if (j.type == "NEW_IMAGE") // sent by server after learning process...
        {
            // console.log("receive new image...");
            // images.push({
            //     hash: j.hash,
            //     identity: j.identity,
            //     image: getDataURLFromRGB(j.content),
            //     representation: j.representation
            // });
            // redrawPeople();

            // var msg = {'type': 'REPLY'};
            // socket.send(JSON.stringify(msg));
            // console.log("reply server having received the image...");
        }
        else if (j.type == "IDENTITIES")
        {
            // var h = "Last updated: " + (new Date()).toTimeString();
            // h += "<ul>";
            // var len = j.identities.length
            // if (len > 0)
            // {
            //     for (var i = 0; i < len; i++)
            //     {
            //         var identity = "Unknown";
            //         var idIdx = j.identities[i];
            //         if (idIdx != -1) {
            //             identity = people[idIdx];
            //         }
            //         h += "<li>" + identity + "</li>";
            //     }
            // } else {
            //     h += "<li>Nobody detected.</li>";
            // }
            // h += "</ul>"
            // $("#peopleInVideo").html(h);

        }

        else if (j.type == "SKIP")
        {
            console.log("Server side still cannot capture image, skip frame...")
        }

        else if (j.type == "ANNOTATED")
        {
            var startTime, endTime;
            startTime = new Date();            
            var serverfps=1;
            var NeedRedraw = false;
            ncount += 1;

            //Update the server load state
            console.log
            if(j.loadingstate == false)
            {
                $("#serverloadstate").html("<font size=5 color=green>Server database is ready!</font>");
            }
            else
            {   var dotnum = "";
                for(var zx=0; zx<ncount%4; zx++)
                {
                    dotnum = dotnum + ".";
                }
                $("#serverloadstate").html("<font size=5 color=red>Server database is updating! Please wait " + dotnum + "</font>");
            }


            if(j.learnmsg.length > 0 )
            {
                console.log("receive new images...");
                for(var gg=0; gg<j.learnmsg.length; gg++ )
                {
                    images.push({
                        hash: j.learnmsg[gg].hash,
                        identity: j.learnmsg[gg].identity,
                        image: getDataURLFromRGB(j.learnmsg[gg].content),
                        representation: j.learnmsg[gg].representation
                    });
                    //console.log("Push one image with identity" + j.learnmsg[gg].identity);
                }
                //console.log(images);
                redrawPeople();
            }
            // $("#detectedFaces").html(
            //     "<img src='" + j['content'] + "' width='430px'></img>"
            // )

            //Given the j['content] = [[name1, bb1], [name2, bb2], ...]
            //Draw the bounding box and the name in the client side in order to speed up

            //j.content == "None" means server cannot get image of that frame, so skip frame
            if(j.content != "None")
            {
                var canvas;
                var vid_now;

  
                // console.log(j.id);
                // if(j.id==0)
                // {
                //     canvas = document.getElementById('drawing');
                // //     vid_now = vid;
                // //     // $("#videoel").html(
                // //     //     "<img src='" + j['image'] + "' width='816' height='459' style='position: absolute; left: 45px; top: 0; z-index: 0;'></img>"
                // //     // )
                // }
                // else
                // {
                //     canvas = document.getElementById('drawing2');
                // //     vid_now = vid2;
                // //     // $("#videoel2").html(
                // //     //     "<img src='" + j['image'] + "' width='816' height='459' style='position: absolute; left: 872px; top: 0; z-index: 0;'></img>"
                // //     // )                
                // }

                // //console.log("debug pt 1");
                // vid_now.src = j.image;
 

                //console.log("debug pt 2");

                // vid_now.width = ui_width;
                // vid_now.height = ui_height;
    
                //canvas.width = vid.width;
                //canvas.height = vid.height;

                //ctx.drawImage(vid, 0, 0, vid.width, vid.height);// draw what the preview looks, in order to clear previous drawing
    
                //console.log("debug pt 3");

                var colorname = ['black', 'green', 'red', 'blue','purple', 'yellow', 'white'];
                var ratio = ui_width/zwidth;
                //var ratio_x = ui_width/zwidth;
                //var ratio_y = ui_height/zheight;

                //clear prev drawing, if necessary
                // if(j.content.length>0 ||IsPrevHasBBs[cam_id]==1)
                // {
                //     for(var x=0; x<j.content.length; x++)
                //     {
                //         cam_id = j.content[x][0];
                //             canvas = document.getElementById('drawing'+cam_id);
                //             var ctx = canvas.getContext('2d');
                //             ctx.clearRect(0, 0, canvas.width, canvas.height);   
                //     }
                // }

                
                for(var x=0; x<2; x++)
                {
                    if(IsPrevHasBBs[x]==1)
                    {
                        canvas = document.getElementById('drawing'+x);
                        var ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);   
                    }

                }         
                
             


                //console.log("debug pt 4");
                //for each camera
                for(var q=0; q<j.content.length; q++)
                {
                    cam_id = j.content[q][0];
                    IsPrevHasBBs[cam_id]=0; //refresh it
                    canvas = document.getElementById('drawing'+cam_id);
                    vid_now = document.getElementById('videoel'+cam_id);
                    var ctx = canvas.getContext('2d');
                    //ctx.clearRect(0, 0, canvas.width, canvas.height);      
                                  
                    //for each BoundingBox
                    for(var z=0; z<j.content[q][1].length; z++)
                    {
                        //console.log("debug pt 5");
                        var idee = j.content[q][1][z][5];
                        if (j.content[q][1][z][9]==true || idee ==-1)
                        {
                            var coloridx = 0;
                            if(j.content[q][1][z][5]>=0)
                            {
                                coloridx = identity_ofppl.findIndex(k => k==idee)+1;
                            }
            
                            //Drawing the rectangle bounding box
                            ctx.beginPath();
                            ctx.lineWidth="5";
                            ctx.strokeStyle= colorname[coloridx%7];
                            var margin = 12;
                            ctx.rect(j.content[q][1][z][1]*ratio-margin,j.content[q][1][z][2]*ratio-margin,
                                j.content[q][1][z][3]*ratio+margin*2,j.content[q][1][z][4]*ratio+margin*2);
                            ctx.stroke();
    
                            // Only draw the name of known person
                            if(idee >= 0)
                            {
                                //Draw the name of the detected person
                                ctx.font = "bold 35pt Calibri";
                                ctx.fillStyle = colorname[coloridx%7];
                                ctx.fillText(j.content[q][1][z][0], j.content[q][1][z][1]*ratio-margin, j.content[q][1][z][2]*ratio-margin-10);
                            }
    
                            IsPrevHasBBs[cam_id]=1;
                        }


        
                        //If it is a newly detect person, add to the result table list
                        if(j.content[q][1][z][6])
                        {
                            console.log("A newly detected person!")

                            // var imgIdx = findImageByIdentity(idee);
                            // var rec_img = 0;
                            // if (idee >= 0) rec_img = images[imgIdx].image
        
                            // var nname = "na";
                            // if (idee >= 0)
                            // {
                            //     var pplidx = identity_ofppl.findIndex(k => k==idee);
                            //     nname = people[pplidx];
                            // }

                            var d = new Date();

                            // var ppcanvas = document.createElement("CANVAS");
                            // ppcanvas.width = display_image_size;
                            // ppcanvas.height = display_image_size;
                            // var ppctx = ppcanvas.getContext('2d');

                            // while(!vid_now.complete)
                            // {
                            //     console.log("Src.img still not finish rendering..., waiting");
                            //     sleep(1);
                            // }

                            // ppctx.drawImage(vid_now, j.content[q][1][z][1]*ratio, j.content[q][1][z][2]*ratio, j.content[q][1][z][3]*ratio, j.content[q][1][z][4]*ratio,
                            //                     0, 0, display_image_size, display_image_size);// draw what the preview looks, in order to clear previous drawing
                            // var dataurl = ppcanvas.toDataURL('image/jpeg');
                            // console.log(dataurl);
        

                            result_table.push({
                                time: d.toUTCString(),
                                simscore: j.content[q][1][z][7],
                                bestmatchidentity: j.content[q][1][z][5],
                                displayimage: j.content[q][1][z][8],
                                recordimageindex: -1,
                                targetname: j.content[q][1][z][0],
                                IsSelected: false,
                                camid: cam_id
                            });

                            ccount += 1;

                            //if(j.content[q][1][z][5]>=0 || ccount % 10 ==0 )    NeedRedraw = true;
                            //if(ccount % 5 ==0 || ncount % 5 ==0)    NeedRedraw = true;
                            NeedRedraw = true;
                            
                        }
        
                    }
                    if(cam_id==1)
                    {
                        //console.log("debug pt 7");
                        //ctx.font = "30pt Calibri";
                        //ctx.fillStyle = 'black';
                        serverfps = j.fps;
                        //ctx.fillText("server fps: " + serverfps + ", client fps: " + (1000.0/timediff).toFixed(2) , 30, 100);
                    }

                    // if(cam_id==1)
                    // {
                    //     //console.log("debug pt 7");
                    //     ctx.font = "30pt Calibri";
                    //     ctx.fillStyle = 'black';
                    //     serverfps = j.fps;
                    //     ctx.fillText("server fps: " + serverfps + ", client fps: " + (1000.0/timediff).toFixed(2) , 30, 100);
                    // }
                }

                vid.src = j.image[0];

                if (j.imageset ==2 )
                {
                    vid2.src = j.image[1];
                }  

                //console.log("debug pt 6");
                //Also print fps at the top left corner

                    

                //Also print the time
                // ctx.font = "30pt Calibri";
                // ctx.fillStyle = 'green';
                // var tt = new Date()
                // ctx.fillText(tt.toString() , 40, 80);   
                
            

                //}
                //console.log("debug pt 8");
            }
            else
            {
                console.log("Server has no processed frame yet...")
            }

            receivetime = new Date();
            timediff = receivetime-sendtime+0.00000001;

            sendtime = new Date();

            // if(counter < 2*defaultNumNulls+1)
            // {
            //     receivedTimes.push(new Date());
            //     sentTimes.push(new Date());
            //     counter++;
            // }

            // if(j.id==0) tok++;
            // else tok2++;
            tok++;

            // if(counter == defaultNumNulls)
            // {
            //     updateRTT_frame();
            // }

            endTime = new Date();
            var time = endTime - startTime; //in ms
            console.log("annotate image takes: " + time + " ms");
            console.log("server fps: " + serverfps + " Client fps:" + (1000.0/timediff).toFixed(2))

            if(NeedRedraw)
            {
                redrawResultTable();
            }


        }
        // else if (j.type == "TSNE_DATA")
        // {
        //     BootstrapDialog.show({
        //         message: "<img src='" + j['content'] + "' width='100%'></img>"
        //     });
        // }
        else {
            console.log("Unrecognized message type: " + j.type);
        }
    }
    socket.onerror = function(e) {
        console.log("Error creating WebSocket connection to " + address);
        console.log(e);
    }
    socket.onclose = function(e) {
        console.log("Websocket is onclose");
        console.log(e);
        if (e.target == socket) {
            $("#serverStatus").html("Disconnected.");
        }
    }
}

function umSuccess(stream)
{

    // if (vid.mozCaptureStream) {
    //     vid.mozSrcObject = stream;
    // } else {
    //     vid.src = (window.URL && window.URL.createObjectURL(stream)) ||
    //         stream;
    // }

    vid.src = "http://172.18.3.20:8080/stream"

    vid.play();
    vidReady = true;
    //sendFrameLoop();
}

function changeDefaultPersonCallback()
{
    if(defaultPerson>=0)
    {
        var idx = identity_ofppl.findIndex(k => k==defaultPerson);
        $("#person_training").html(people[idx]);
    }
    else
    {
        $("#person_training").html("N.A.");
    }
}

// function changeMergingPersonCallback()
// {
//     if(MergingPerson>=0)
//     {
//         var idx = identity_ofppl.findIndex(k => k==MergingPerson);
//         $("#person_merge").html(people[idx]);
//     }
//     else
//     {
//         $("#person_merge").html("N.A.");
//     }
// }

function addPersonCallback(el) {
    //defaultPerson = people.length;
    var newPerson = $("#addPersonTxt").val();
    if (newPerson == "")
    {
        alert("Please enter the person name!")
        return;
    }
    people.push(newPerson);
    var newidentity = 0;
    if (identity_ofppl.length>0)
    {
        newidentity = Math.max(...identity_ofppl)+1;
    }
    identity_ofppl.push(newidentity);
    defaultPerson = newidentity

    $("#addPersonTxt").val("");

    if (socket != null) {
        var msg = {
            'type': 'ADD_PERSON',
            'val': newPerson,
            'ide': newidentity
        };
        socket.send(JSON.stringify(msg));
    }
    redrawPeople();
    changeDefaultPersonCallback();
}

function deletePerson(ide)
{
    var idx = identity_ofppl.findIndex(k => k==ide);
    var pname = people[idx];
    people.splice(idx, 1);
    identity_ofppl.splice(idx, 1);

    var imgidx = findImageByIdentity(ide);
    while(imgidx >= 0)
    {
        images.splice(imgidx, 1)
        imgidx = findImageByIdentity(ide);
    }

    if (socket != null) {
        var msg = {
            'type': 'DELETE_PERSON',
            'val': ide,
            'name': pname
        };
        socket.send(JSON.stringify(msg));
    }

    redrawPeople();

    if(ide == defaultPerson)
    {
        defaultPerson = -1;
        changeDefaultPersonCallback();
    }
}

function deletePersonCallback(el) {
    deletePerson(defaultPerson);
}

function openPersonImages(idx)
{
    var modal = document.getElementById('myModal');
    modal.style.display = "block";

    var hd_counter =0;
    var num_of_col = 13;    
    var hd = "<font size=5>Learnt images of the person</font> <br><br>";
    hd += "<table border='1'>";

    for(var p =0; p<images.length; p++)
    {
        if (images[p].identity == idx)
        {
            if(hd_counter%num_of_col == 0)
            {
                hd += "<tr>";
            }

            hd += "<td>";
            hd += ( "<img src=" + images[p].image + "> ");
            hd += "<a href='javascript:removeImageInWindow(" + idx + ", " + images[p].hash + ")'>(X)</a>";
            hd += "</td>";

            if(hd_counter%num_of_col == num_of_col-1)
            {
                hd += "</tr>";
            }            

            hd_counter ++;
        }
    }
    hd += "</table><br>";

    $("#modalc").html(hd);
}

function OpenLearnPersonWindow()
{
    var hd_counter =0;
    var num_of_col = 8;
    var stun_result_table_length = result_table.length;

    var unlearn_num =0;
    for(var pp =stun_result_table_length-1; pp>=0; pp--)
    {
        if (result_table[pp].IsSelected == false)
        {    
            unlearn_num += 1;
        }
    }

    var hd = "<font size=5>Unknown person list (" + unlearn_num + ")</font> <br><br>";

    hd += "Name: <input type='text' id='learn_name' size=15> or Merge with existing person: ";
    hd += "<select id='MergeExisting'><option value=-1>N.A.</option>";

    for(var q=0; q<people.length; q++)
    {
        hd += "<option value=" + identity_ofppl[q] + ">" + people[q] + "(" + info[q] + ") </option>";
    }

    hd += "</select><br>";

    hd += "<input type='checkbox' id='LearnWithAug'> Learn with augmentation<br>";
    hd += "<button type='button' onclick='Select_All_to_Learn(" + stun_result_table_length + ")'>Select All</button> "
    hd += "<button type='button' onclick='UnSelect_All_to_Learn(" + stun_result_table_length + ")'>Unselect All</button> "
    hd += "<button type='button' onclick='Hide_Selected_Person(" + stun_result_table_length + ")'>Hide selected person</button> "
    hd += "<button type='button' onclick='learnPersonCallback(" + stun_result_table_length + ")'>Learn selected unknown person</button> <br>";
    hd += "<table border='1'>";

    for(var p =stun_result_table_length-1; p>=0; p--)
    {
        if (hd_counter <= unmatched_result_max_show && result_table[p].IsSelected == false)
        {
            if(hd_counter%num_of_col == 0)
            {
                hd += "<tr>";
            }

            hd += "<td>";
            hd += ( "<img src=" + result_table[p].displayimage + "> ");
            hd += "<input type='checkbox' id='result_table_" + p + "' value='" + p + "' style='height:25px; width:25px; vertical-align: middle;'>"
            hd += "<br>" + (result_table[p].time);
            hd += "<br> <b>[Cam: " + result_table[p].camid + "]</b>";
            hd += "</td>";

            if(hd_counter%num_of_col == num_of_col-1)
            {
                hd += "<td> Entire row";
                hd += "<input type='checkbox' onchange='DoEntireRow(" + num_of_col + ", " + p + ", " + stun_result_table_length + ")' style='height:25px; width:25px; vertical-align: middle;'>"
                hd += "</td>";
                hd += "</tr>";
            }            

            hd_counter ++;
        }
    }
    hd += "</table><br>";


    hd += "<button type='button' onclick='learnPersonCallback(" + stun_result_table_length + ")'>Learn selected unknown person</button>";

    $("#modalc").html(hd);
    
}

function DoEntireRow(Num_Of_Col, ending_index, stun_length)
{
    var counter = 0;
    for(var p=ending_index; p<stun_length; p++)
    {
        if(counter >= Num_Of_Col) break;
        if(result_table[p].IsSelected == false)
        {
            counter++;
            var radio = document.getElementById('result_table_'+p);
            radio.checked = !radio.checked;  
        }
    }    
}

function Hide_Selected_Person(stun_length)
{
    for(var p=0; p<stun_length; p++)
    {
        var radio = document.getElementById('result_table_'+p);
        if (result_table[p].IsSelected == false && radio.checked)
        {
            result_table[p].IsSelected = true;
        }
    }    
    OpenLearnPersonWindow();
}

function Select_All_to_Learn(stun_length)
{
    for(var p=0; p<stun_length; p++)
    {
        if(result_table[p].IsSelected == false)
        {
            var radio = document.getElementById('result_table_'+p);
            radio.checked = true;  
        }
    }
}

function UnSelect_All_to_Learn(stun_length)
{
    for(var p=0; p<stun_length; p++)
    {
        if(result_table[p].IsSelected == false)
        {
            var radio = document.getElementById('result_table_'+p);
            radio.checked = false;  
        }  
    }
}

function learnPersonCallback(stun_length) {
    var learning_img_list_dataurl= [];
    for(var p=0; p<stun_length-1; p++)
    {
        var radio = document.getElementById('result_table_'+p);
        if(result_table[p].IsSelected == false && radio.checked)
        {
            learning_img_list_dataurl.push(result_table[p].displayimage);
        }        
    }

    if(learning_img_list_dataurl.length==0)
    {
        alert("At least select one unknown photo to learn!!!");
        return;
    }

    var e = document.getElementById('MergeExisting');
    var MexistingID = parseInt(e.options[e.selectedIndex].value);


    var TargetPersonName = $("#learn_name").val();
    if(TargetPersonName == "" && MexistingID == -1)
    {
        alert("Please enter a name for the unknown photo to learn!!!");
        return;
    }
    
    var IsLearnWithAug = false;
    if(document.getElementById('LearnWithAug').checked)
    {
        IsLearnWithAug = true;
    }


    for(var p=0; p<stun_length-1; p++)
    {
        var radio = document.getElementById('result_table_'+p);
        if(result_table[p].IsSelected == false && radio.checked)
        {
            result_table[p].IsSelected = true;
        }        
    }

    learnmsg = {
            'type': 'LEARN_UNKNOWN_PERSON',
            'img_list_dataurl': learning_img_list_dataurl,
            'name': TargetPersonName,
            'IsAug': IsLearnWithAug,
            'MergeExisting': MexistingID};
    console.log("Sent out learn images...");
    //console.log(learning_img_list_dataurl);

    // people.splice(idx2, 1);
    // identity_ofppl.splice(idx2, 1);

    //redrawPeople();

    // MergingPerson = -1;
    // changeMergingPersonCallback();

    //Close the modal after finish
    var modal = document.getElementById('myModal');
    modal.style.display = "none";
}

function mergePersonCallback(el) {
    var checked_list= [];
    for(var e=0; e<people.length; e++)
    {
        var radio = document.getElementById('m'+identity_ofppl[e]);
        if(radio.checked)
        {
            checked_list.push(identity_ofppl[e]);
        }
    }

    if(checked_list.length<2)
    {
        alert("At least merge between 2 people!!!");
        return;
    }

    var idx1 = identity_ofppl.findIndex(k => k==checked_list[0]);
    for(var f=1; f<checked_list.length;f++)
    {
        var idx2 = identity_ofppl.findIndex(k => k==checked_list[f]);

        var imgIdx = 0;
        var len = images.length;
        for (imgIdx = 0; imgIdx < len; imgIdx++) {
            if (images[imgIdx].identity == checked_list[f])
            {
                images[imgIdx].identity = checked_list[0];
            }
        }

        if (socket != null) {
            var msg = {
                'type': 'MERGE_PERSON',
                'master': checked_list[0],
                'slave': checked_list[f],
                'master_name': people[idx1],
                'slave_name': people[idx2]
            };
            socket.send(JSON.stringify(msg));
        }

        people.splice(idx2, 1);
        identity_ofppl.splice(idx2, 1);
    }


    redrawPeople();

    // MergingPerson = -1;
    // changeMergingPersonCallback();

}

function editPersonCallback(el) {
    var TargetPersonName = $("#editPersonTxt").val();
    if (TargetPersonName == "")
    {
        alert("Please enter the person name!")
        return;
    }
    var idx = identity_ofppl.findIndex(k => k==defaultPerson);
    people[idx] = (TargetPersonName);
    $("#editPersonTxt").val("");

    if (socket != null) {
        var msg = {
            'type': 'EDIT_PERSON_NAME',
            'val': TargetPersonName,
            'identity': defaultPerson
        };
        socket.send(JSON.stringify(msg));
    }
    redrawPeople();
    changeDefaultPersonCallback();
}

function trainingChkCallback() {
    training = $("#trainingChk").prop('checked');
    // if (training) {
    //     makeTabActive("tab-preview");
    // } else {
    //     makeTabActive("tab-annotated");
    // }
    if (socket != null) {
        var msg = {
            'type': 'TRAINING',
            'val': training
        };
        socket.send(JSON.stringify(msg));
    }

    if(!training)
    {
        alert("Training finish, please wait a few seconds for the database to refresh...")
    }
}

function findImageByHash(hash) {
    var imgIdx = 0;
    var len = images.length;
    for (imgIdx = 0; imgIdx < len; imgIdx++) {
        if (images[imgIdx].hash == hash) {
            //console.log("  + Image found.");
            return imgIdx;
        }
    }
    return -1;
}

function findImageByIdentity(identity, skip=0) {
    var imgIdx = 0;
    var len = images.length;
    var yescount = 0;
    for (imgIdx = 0; imgIdx < len; imgIdx++) {
        if (images[imgIdx].identity == identity)
        {
            if( yescount >= skip)
            {
                //console.log("  + Image found.");
                return imgIdx;
            }
            else yescount ++;
        }
    }
    return -1;
}

//When clicking the radio button to change identity of the wrongly labeled person
function updateIdentity(hash, idx)
{
    console.log("updateIdentity hash:" + hash + "idx: " + idx);
    var imgIdx = findImageByHash(hash);
    if (imgIdx >= 0) {
        images[imgIdx].identity = idx;
        var msg = {
            'type': 'UPDATE_IDENTITY',
            'hash': hash,
            'idx': idx
        };
        socket.send(JSON.stringify(msg));
    }
    redrawPeople();
}

function removeImageInWindow(idx, hash)
{
    removeImage("'" + hash + "'");
    openPersonImages(idx);
}

function removeImage(hash)
{
    console.log("Removing " + hash);
    var imgIdx = findImageByHash(hash);
    if (imgIdx >= 0)
    {
        var ide = images[imgIdx].identity
        images.splice(imgIdx, 1);
        redrawPeople();
        var msg = {
            'type': 'REMOVE_IMAGE',
            'hash': hash,
            'ide': ide
        };
        socket.send(JSON.stringify(msg));
    }
    redrawPeople();
}

function Augment_All_Existing()
{
    aug_all_existing = 1;
}
