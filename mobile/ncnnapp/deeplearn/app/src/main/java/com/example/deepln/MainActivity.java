package com.example.deepln;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;

import com.camera.styletransfer;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import android.util.Log;
import android.view.View;

import androidx.core.view.GravityCompat;
import androidx.appcompat.app.ActionBarDrawerToggle;

import android.view.MenuItem;

import com.google.android.material.navigation.NavigationView;

import androidx.drawerlayout.widget.DrawerLayout;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;

import android.view.Menu;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity
        implements NavigationView.OnNavigationItemSelectedListener,cameraFragment.OnFragmentInteractionListener{

    private static final int GET_PERMISSION = 520;
    private static styletransfer transfer;

    private FragmentManager fm;
    private cameraFragment camerafrag;

    private String[] modelPath;
    private styletransfer stransfer;
    private boolean isInit=false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        FloatingActionButton fab = findViewById(R.id.fab);

        //====
        transfer=new styletransfer();

        fm= getSupportFragmentManager();

        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, transfer.stringFromJNI(), Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
            }
        });
        DrawerLayout drawer = findViewById(R.id.drawer_layout);
        NavigationView navigationView = findViewById(R.id.nav_view);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(
                this, drawer, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawer.addDrawerListener(toggle);
        toggle.syncState();
        navigationView.setNavigationItemSelectedListener(this);

        final Activity act=this;
        new Thread() {
            public void run() {
                // 1.执行耗时操作
                //String[] models=new String[]{"candy.bin", "mosaic.bin", "pointilism.bin", "rain_princess.bin", "udnie.bin"};
                String[] models=new String[]{"candy.bin", "mosaic.bin"};
                modelPath=new String[models.length];

                for(int i=0;i<models.length;i++){
                    modelPath[i]=getAssetsCacheFile(act,models[i]);
                    Log.d("asserts path", "model path:"+modelPath[i]);
                }

                stransfer=new styletransfer();
                stransfer.InitEnv(null,modelPath);
                isInit=true;

                for(int i=0;i<modelPath.length;i++){
                    //清除缓存
                    File mofile=new File(modelPath[i]);
                    if(mofile.exists()){
                        Log.d("remove model data", "model path:"+mofile.delete());

                    }
                }

                runOnUiThread(new Runnable(){
                    @Override
                    public void run() {
                        // 2.更新UI
                        Log.d("asserts path", "Load model finished, enjoy!");
                       // Toast.makeText(act, "Load model finished, enjoy!", Toast.LENGTH_LONG).show();
                    }
                });
            }
        }.start();
    }


    public styletransfer gotTransfer(){
        if(isInit){
            return stransfer;
        }
        return null;
    }

    public String[] gotModelPath(){
        if(modelPath.length==0){
            Log.d("gotModelPath", "modelpaths is empty");
        }
        return modelPath;
    }

    @Override
    public void onBackPressed() {
        DrawerLayout drawer = findViewById(R.id.drawer_layout);
        if (drawer.isDrawerOpen(GravityCompat.START)) {
            drawer.closeDrawer(GravityCompat.START);
        } else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == GET_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                //
                Toast.makeText(this, "Permission accept", Toast.LENGTH_SHORT).show();
            } else {
                // 如果检查到没有被授权则应该到手机setting里面设置授权，如果未显示授权项则检查是否在AndroidManifest.xml中配置了相应权限，否则删除手机上的app重新部署检查；
                Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
            }
            return;
        }

    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @SuppressWarnings("StatementWithEmptyBody")
    @Override
    public boolean onNavigationItemSelected(MenuItem item) {
        // Handle navigation view item clicks here.
            int id = item.getItemId();
            FragmentTransaction ft= fm.beginTransaction();
            hideFragment(ft);
            try{
                Log.d("onNavigationItemSelected", "id:"+id+"nva:"+R.id.nav_home);
                if (id == R.id.nav_home) {
                    // Handle the action
                    if (camerafrag==null){
                        camerafrag=new cameraFragment();
                        ft.add(R.id.ftshow,camerafrag);

                    }else{
                        ft.show(camerafrag);
                    }

                } else if (id == R.id.nav_gallery) {


                } else if (id == R.id.nav_slideshow) {


                } else if (id == R.id.nav_tools) {

                }

                DrawerLayout drawer = findViewById(R.id.drawer_layout);
                drawer.closeDrawer(GravityCompat.START);

                //add by clause
                ft.commit();
            }catch (Exception e){
                e.printStackTrace();
            }
            return true;
    }

    private void hideFragment(FragmentTransaction ft) {
        if (camerafrag != null) {
            camerafrag.onHide();
            ft.hide(camerafrag);
        }

    }

    public String getAssetsCacheFile(Activity activity, String fileName) {

        File cacheFile = new File(activity.getCacheDir(), fileName);
        try {
            //InputStream inputStream = this.getAssets().open(fileName);
            InputStream inputStream=activity.getClass().getClassLoader().getResourceAsStream ("assets/"+fileName);
            try {
                FileOutputStream outputStream = new FileOutputStream(cacheFile);
                try {
                    byte[] buf = new byte[1024];
                    int total=0;
                    int len;
                    while ((len = inputStream.read(buf)) > 0) {
                        outputStream.write(buf, 0, len);
                        outputStream.flush();
                        total+=len;
                    }

                    Log.d("getAssetsCacheFile", fileName+",total bytes:"+total);
                } finally {
                    outputStream.close();
                }
            } finally {
                inputStream.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return cacheFile.getAbsolutePath();
    }
}
