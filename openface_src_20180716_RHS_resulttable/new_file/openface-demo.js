/*
Copyright 2015-2016 Carnegie Mellon University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
$.fn.pressEnter = function(fn) {

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

var zwidth = 448; //640; //800
var zheight = 252; //360; // 450
var display_image_size = 48;

function sendFrameLoop()
{
    if (socket == null || socket.readyState != socket.OPEN ||
        !vidReady || numNulls != defaultNumNulls) {
        return;
    }

    if (tok > 0)
    {
        var startTime, endTime;
        startTime = new Date();

        var canvas = document.createElement('canvas');
        canvas.width = zwidth;
        canvas.height = zheight;
        var cc = canvas.getContext('2d');
        cc.drawImage(vid, 0, 0, zwidth, zheight);
        var apx = cc.getImageData(0, 0, zwidth, zheight);
        
        // 2nd argument is image quality, from 1 (full quality) to 0 (zero quality), default 0.92
        var dataURL = canvas.toDataURL('image/jpeg')

        var msg = {
            'type': 'FRAME',
            'dataURL': dataURL,
            'identity': defaultPerson
        };
        socket.send(JSON.stringify(msg));
        tok--;

        endTime = new Date();
        var timeDiff = endTime - startTime; //in ms
        console.log("sendFrameLoop takes: " + timeDiff + " ms");
    }
    setTimeout(function() {requestAnimFrame(sendFrameLoop)}, 250);
}


function getPeopleInfoHtml() {
    var info = {'-1': 0};
    var len = people.length;
    for (var i = 0; i < len; i++) {
        info[i] = 0;
    }

    var len = images.length;
    for (var i = 0; i < len; i++) {
        id = images[i].identity;
        info[id] += 1;
    }

    var h = "<li><b>Unknown:</b> "+info['-1']+"</li>";
    var len = people.length;
    for (var i = 0; i < len; i++) {
        h += "<li style='color:red;font-size:25px'><b>"+people[i]+":</b> "+info[i]+"</li>";
    }
    return h;
}

//Redraw the list of added person at the bottom of the page based on var people and images
function redrawPeople() {
    var startTime, endTime;
    startTime = new Date();
    var context = {people: people, images: images};
    $("#peopleTable").html(peopleTableTmpl(context));

    var context = {people: people};
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

    var h ="<table border='1'>";
    for(var p =result_table.length-1; p>=0; p--)
    {
        h += "<tr><td>";
        h += ( "<img src=" + result_table[p].displayimage + ">");
        h += "</td><td>";

        if(result_table[p].bestmatchidentity>=0)
        {
            var imgIdx = findImageByIdentity(result_table[p].bestmatchidentity);      
            h += ( "<img src=" + images[imgIdx].image + ">");
        }
        else
        {
            h += "Unknown"
        }
        
        h += "</td></tr><tr>";

        h += "<tr><td colspan='2'>"

        if(result_table[p].bestmatchidentity>=0)
        {
            h += "<font color=green>Matched</font><br>"
        }
        else
        {
            h += "<font color=red>Not matched!</font><br>"
        }

        h += (result_table[p].time + "<br>");
        h += ("Name: " +  people[result_table[p].bestmatchidentity] + "<br>");

        if(result_table[p].bestmatchidentity>=0)
        {
            h += ("similarity score: " +  result_table[p].simscore + "<br>");
        }
        h += "</td><tr>";
    }
    h += "</table>";
    $("#result_table").html(h);

    endTime = new Date();
    var timeDiff = endTime - startTime; //in ms
    console.log("redrawResultTable() takes: " + timeDiff + " ms");    
}

function getDataURLFromRGB(rgb) {
    var rgbLen = rgb.length;

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
function sendState() {
    var msg = {
        'type': 'ALL_STATE',
        'images': images,
        'people': people,
        'training': training
    };
    socket.send(JSON.stringify(msg));
}

var counter=0;
function createSocket(address, name) {
    socket = new WebSocket(address);
    socketName = name;
    socket.binaryType = "arraybuffer";
    socket.onopen = function() {
        $("#serverStatus").html("Connected to " + name);
        sentTimes = [];
        receivedTimes = [];
        tok = defaultTok;
        numNulls = 0

        socket.send(JSON.stringify({'type': 'NULL'}));
        sentTimes.push(new Date());
    }
    socket.onmessage = function(e) {
        console.log(e);
        j = JSON.parse(e.data)
        if (j.type == "NULL") {
            receivedTimes.push(new Date());
            numNulls++;
            if (numNulls == defaultNumNulls) {
                updateRTT_null();
                sendState();
                sendFrameLoop();
            } else {
                socket.send(JSON.stringify({'type': 'NULL'}));
            }
            sentTimes.push(new Date());
        } else if (j.type == "PROCESSED")
        {
            receivedTimes.push(new Date());
            sentTimes.push(new Date());
            tok++; 
            counter++;
            if(counter == defaultNumNulls)
            {
                updateRTT_frame();
            }
        }
        else if (j.type == "NEW_IMAGE") {
            images.push({
                hash: j.hash,
                identity: j.identity,
                image: getDataURLFromRGB(j.content),
                representation: j.representation
            });
            redrawPeople();
        } else if (j.type == "IDENTITIES") {
            var h = "Last updated: " + (new Date()).toTimeString();
            h += "<ul>";
            var len = j.identities.length
            if (len > 0) {
                for (var i = 0; i < len; i++) {
                    var identity = "Unknown";
                    var idIdx = j.identities[i];
                    if (idIdx != -1) {
                        identity = people[idIdx];
                    }
                    h += "<li>" + identity + "</li>";
                }
            } else {
                h += "<li>Nobody detected.</li>";
            }
            h += "</ul>"
            $("#peopleInVideo").html(h);

        }

        else if (j.type == "ANNOTATED")
        {
            var startTime, endTime;
            startTime = new Date();
            // $("#detectedFaces").html(
            //     "<img src='" + j['content'] + "' width='430px'></img>"
            // )

            //Given the j['content] = [[name1, bb1], [name2, bb2], ...]
            //Draw the bounding box and the name in the client side in order to speed up
            
            var canvas = document.getElementById('detectedFaces');
            canvas.width = vid.width;
            canvas.height = vid.height;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(vid, 0, 0, vid.width, vid.height);// draw what the preview looks, in order to clear previous drawing

            var ratio = 1280.0/zwidth;
            for(var z=0; z<j.content.length; z++)
            {
                //Drawing the rectangle bounding box
                ctx.beginPath();
                ctx.lineWidth="5";
                ctx.strokeStyle="green";
                ctx.rect(j.content[z][1]*ratio,j.content[z][2]*ratio,j.content[z][3]*ratio,j.content[z][4]*ratio); 
                ctx.stroke();       
                
                //Draw the name of the detected person
                ctx.font = "30pt Calibri";
                ctx.fillText(j.content[z][0], j.content[z][1]*ratio, j.content[z][2]*ratio-10);

                //If it is a newly detect person, add to the result table list
                if(j.content[z][6])
                {
                    var ppcanvas = document.createElement("CANVAS");
                    ppcanvas.width = display_image_size;
                    ppcanvas.height = display_image_size;
                    var ppctx = ppcanvas.getContext('2d');
                    ppctx.drawImage(vid, j.content[z][1]*ratio, j.content[z][2]*ratio, j.content[z][3]*ratio, j.content[z][4]*ratio,
                                        0, 0, display_image_size, display_image_size);// draw what the preview looks, in order to clear previous drawing
                    var dataurl = ppcanvas.toDataURL('image/jpeg');
                    console.log(dataurl);

                    var d = new Date();
                    result_table.push({
                        time: d.toUTCString(),
                        simscore: j.content[z][7],
                        bestmatchidentity: j.content[z][5],
                        displayimage: dataurl
                    });
                    redrawResultTable();  
                }
              
            }

            //Also print fps at the top left corner
            ctx.font = "30pt Calibri";
            ctx.fillText("fps: " + j.fps, 40, 40);

            if(counter < 2*defaultNumNulls+1)
            {
                receivedTimes.push(new Date());
                sentTimes.push(new Date());
                counter++;
            }

            tok++;
            if(counter == defaultNumNulls)
            {
                updateRTT_frame();
            }

            endTime = new Date();
            var timeDiff = endTime - startTime; //in ms
            console.log("annotate image takes: " + timeDiff + " ms");            


        } else if (j.type == "TSNE_DATA") {
            BootstrapDialog.show({
                message: "<img src='" + j['content'] + "' width='100%'></img>"
            });
        } else {
            console.log("Unrecognized message type: " + j.type);
        }
    }
    socket.onerror = function(e) {
        console.log("Error creating WebSocket connection to " + address);
        console.log(e);
    }
    socket.onclose = function(e) {
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
    sendFrameLoop();
}

function changeDefaultPersonCallback()
{
    if(defaultPerson>=0)
    {
        $("#person_training").html(people[defaultPerson]);
    }
    else
    {
        $("#person_training").html("N.A.");
    }
}

function addPersonCallback(el) {
    defaultPerson = people.length;
    var newPerson = $("#addPersonTxt").val();
    if (newPerson == "")
    {
        alert("Please enter the person name!")
        return;
    } 
    people.push(newPerson);
    $("#addPersonTxt").val("");

    if (socket != null) {
        var msg = {
            'type': 'ADD_PERSON',
            'val': newPerson
        };
        socket.send(JSON.stringify(msg));
    }
    redrawPeople();
    changeDefaultPersonCallback();
}

function editPersonCallback(el) {
    var TargetPersonName = $("#editPersonTxt").val();
    if (TargetPersonName == "")
    {
        alert("Please enter the person name!")
        return;
    } 
    people[defaultPerson] = (TargetPersonName);
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

// function viewTSNECallback(el) {
//     if (socket != null) {
//         var msg = {
//             'type': 'REQ_TSNE',
//             'people': people
//         };
//         socket.send(JSON.stringify(msg));
//     }
// }

function findImageByHash(hash) {
    var imgIdx = 0;
    var len = images.length;
    for (imgIdx = 0; imgIdx < len; imgIdx++) {
        if (images[imgIdx].hash == hash) {
            console.log("  + Image found.");
            return imgIdx;
        }
    }
    return -1;
}

function findImageByIdentity(identity) {
    var imgIdx = 0;
    var len = images.length;
    for (imgIdx = 0; imgIdx < len; imgIdx++) {
        if (images[imgIdx].identity == identity) {
            console.log("  + Image found.");
            return imgIdx;
        }
    }
    return -1;
}

//When clicking the radio button to change identity of the wrongly labeled person
function updateIdentity(hash, idx) {
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
}

function removeImage(hash) {
    console.log("Removing " + hash);
    var imgIdx = findImageByHash(hash);
    if (imgIdx >= 0) {
        images.splice(imgIdx, 1);
        redrawPeople();
        var msg = {
            'type': 'REMOVE_IMAGE',
            'hash': hash
        };
        socket.send(JSON.stringify(msg));
    }
}

// function changeServerCallback() {
//     $(this).addClass("active").siblings().removeClass("active");
//     switch ($(this).html()) {
//     case "Local":
//         socket.close();
//         redrawPeople();
//         createSocket("wss:" + window.location.hostname + ":9000", "Local");
//         break;
//     case "CMU":
//         socket.close();
//         redrawPeople();
//         createSocket("wss://facerec.cmusatyalab.org:9000", "CMU");
//         break;
//     case "AWS East":
//         socket.close();
//         redrawPeople();
//         createSocket("wss://54.159.128.49:9000", "AWS-East");
//         break;
//     case "AWS West":
//         socket.close();
//         redrawPeople();
//         createSocket("wss://54.188.234.61:9000", "AWS-West");
//         break;
//     default:
//         alert("Unrecognized server: " + $(this.html()));
//     }
// }
